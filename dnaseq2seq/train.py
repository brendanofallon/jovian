
import logging
from collections import defaultdict

import yaml
from datetime import datetime
import os
from pygit2 import Repository

import numpy as np
import torch
from torch import nn


logger = logging.getLogger(__name__)

import vcf
import loader
import util
from model import VarTransformer
from swloss import SmithWatermanLoss

ENABLE_WANDB = os.getenv('ENABLE_WANDB', False)

if ENABLE_WANDB:
    import wandb


DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


MODEL_DTYPE=torch.float16

class TrainLogger:
    """ Simple utility for writing various items to a log file CSV """

    def __init__(self, output, headers):
        self.headers = list(headers)
        if type(output) == str:
            self.output = open(output, "a")
        else:
            self.output = output
        self._write_header()


    def _write_header(self):
        self.output.write(",".join(self.headers) + "\n")
        self._flush_and_fsync()

    def _flush_and_fsync(self):
        try:
            self.output.flush()
            os.fsync()
        except:
            pass

    def log(self, items):
        assert len(items) == len(self.headers), f"Expected {len(self.headers)} items to log, but got {len(items)}"
        self.output.write(
            ",".join(str(items[k]) for k in self.headers) + "\n"
        )
        self._flush_and_fsync()


def calc_time_sums(
        time_sums={},
        decomp_time=0.0,
        start=None,
        decomp_and_load=None,
        zero_grad=None,
        forward_pass=None,
        loss=None,
        midmatch=None,
        backward_pass=None,
        optimize=None,
):
    load_time = (decomp_and_load - start).total_seconds() - decomp_time
    return dict(
        decomp_time = decomp_time + time_sums.get("decomp_time", 0.0),
        load_time = load_time + time_sums.get("load_time", 0.0),
        batch_count= 1 + time_sums.get("batch_count", 0),
        batch_time=(optimize - start).total_seconds() + time_sums.get("batch_time", 0.0),
        zero_grad_time=(zero_grad - decomp_and_load).total_seconds() + time_sums.get("zero_grad_time", 0.0),
        forward_pass_time=(forward_pass - zero_grad).total_seconds() + time_sums.get("forward_pass_time", 0.0),
        loss_time=(loss - forward_pass).total_seconds() + time_sums.get("loss_time", 0.0),
        midmatch_time=(midmatch - loss).total_seconds() + time_sums.get("midmatch_time", 0.0),
        backward_pass_time=(backward_pass - midmatch).total_seconds() + time_sums.get("backward_pass_time", 0.0),
        optimize_time=(optimize - backward_pass).total_seconds() + time_sums.get("optimize_time", 0.0),
        train_time=(optimize - zero_grad).total_seconds() + time_sums.get("train_time", 0.0)
    )

def compute_twohap_loss(preds, tgt, criterion):
    """
    Iterate over every item in the batch, and compute the loss in both configurations (under torch.no_grad())
    then swap haplotypes (dimension index 1) in the predictions if that leads to a lower loss
    Finally, re-compute loss with the new configuration for all samples and return it, storing gradients this time
    """
    # Compute losses in both configurations, and use the best
    with torch.no_grad():
        for b in range(preds.shape[0]):
            loss1 = criterion(preds[b, :, :, :].flatten(start_dim=0, end_dim=1),
                              tgt[b, :, :].flatten())
            loss2 = criterion(preds[b, :, :, :].flatten(start_dim=0, end_dim=1),
                              tgt[b, torch.tensor([1, 0]), :].flatten())

            if loss2 < loss1:
                preds[b, :, :, :] = preds[b, torch.tensor([1, 0]), :]

    return criterion(preds.flatten(start_dim=0, end_dim=2), tgt.flatten())



def train_epoch(model, optimizer, criterion, loader, batch_size):
    """
    Train for one epoch, which is defined by the loader but usually involves one pass over all input samples
    :param model: Model to train
    :param optimizer: Optimizer to update params
    :param criterion: Loss function
    :param loader: Provides training data
    :param batch_size:
    :return: Sum of losses over each batch, plus fraction of matching bases for ref and alt seq
    """
    model.train()
    epoch_loss_sum = None
    prev_epoch_loss = None
    count = 0
    # init time usage to zero
    epoch_times = {}
    start_time = datetime.now()

    truncate_tgt_len = 148
    for batch, (src, tgt_kmers, tgtvaf, altmask, log_info) in enumerate(loader.iter_once(batch_size)):
        if log_info:
            decomp_time = log_info.get("decomp_time", 0.0)
        else:
            decomp_time = 0.0

        #tgt_kmers = util.tgt_to_kmers(tgt_seq[:, :, 0:truncate_tgt_len]).float().to(DEVICE)
        tgt_kmer_idx = torch.argmax(tgt_kmers, dim=-1)
        tgt_kmers_input = tgt_kmers[:, :, :-1]
        tgt_expected = tgt_kmer_idx[:, :, 1:]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_kmers_input.shape[-2]).to(DEVICE)
        times = dict(start=start_time, decomp_and_load=datetime.now(), decomp_time=decomp_time)

        optimizer.zero_grad()
        times["zero_grad"] = datetime.now()
        seq_preds = model(src, tgt_kmers_input, tgt_mask)
        times["forward_pass"] = datetime.now()

        loss = compute_twohap_loss(seq_preds.float(), tgt_expected, criterion)


        times["loss"] = datetime.now()

        count += 1
        if count % 100 == 0:
            if prev_epoch_loss:
                lossdif = epoch_loss_sum - prev_epoch_loss
            else:
                lossdif = 0
            logger.info(f"Batch {count} : epoch_loss_sum: {epoch_loss_sum:.3f} epoch loss dif: {lossdif:.3f}")

        times["midmatch"] = datetime.now()
        loss.backward()
        times["backward_pass"] = datetime.now()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Not sure what is reasonable here, but we want to prevent the gradient from getting too big
        optimizer.step()
        times["optimize"] = datetime.now()

        if epoch_loss_sum is None:
            epoch_loss_sum = loss.detach().item()
        else:
            prev_epoch_loss = epoch_loss_sum
            epoch_loss_sum += loss.detach().item()
        if np.isnan(epoch_loss_sum):
            logger.warning(f"Loss is NAN!!")
        if batch % 10 == 0:
            logger.info(f"Batch {batch} : loss: {loss.item():.3f}")
            
        #logger.info(f"batch: {batch} loss: {loss.item()} vafloss: {vafloss.item()}")
        epoch_times = calc_time_sums(time_sums=epoch_times, **times)
        start_time = datetime.now()  # reset timer for next batch
        # del src
    # torch.cuda.empty_cache()
    logger.info(f"Trained {batch+1} batches in total for epoch")

    return epoch_loss_sum, epoch_times


def add_result_dicts(src, add):
    for key, subdict in add.items():
        for subkey, value in subdict.items():
            src[key][subkey] += value
    return src


def _calc_hap_accuracy(src, seq_preds, tgt, result_totals):
    # Compute val accuracy
    match = (torch.argmax(seq_preds[:, :, :].flatten(start_dim=0, end_dim=1),
                             dim=1) == tgt[:, :].flatten()
                ).float().mean()

    var_count = 0

    for b in range(src.shape[0]):
        predstr = util.kmer_preds_to_seq(seq_preds[b, :, 0:util.KMER_COUNT], util.i2s)
        tgtstr = util.kmer_idx_to_str(tgt[b, :], util.i2s)
        vc = len(list(vcf.aln_to_vars(tgtstr, predstr)))
        var_count += vc

        # Get TP, FN and FN based on reference, alt and predicted sequence.
        vartype_count = eval_prediction(util.readstr(src[b, :, 0, :]), tgtstr, seq_preds[b, :, 0:util.KMER_COUNT], counts=init_count_dict())
        result_totals = add_result_dicts(result_totals, vartype_count)

    return match, var_count, result_totals


def eval_prediction(refseqstr, altseq, predictions, counts):
    """
    Given a target sequence and two predicted sequences, attempt to determine if the correct *Variants* are
    detected from the target. This uses the vcf.align_seqs(seq1, seq2) method to convert string sequences to Variant
    objects, then compares variant objects
    :param tgt:
    :param predictions:
    :param midwidth:
    :return: Sets of TP, FP, and FN vars
    """
    known_vars = []
    for v in vcf.aln_to_vars(refseqstr, altseq):
        known_vars.append(v)

    pred_vars = []
    predstr = util.kmer_preds_to_seq(predictions[:, 0:util.KMER_COUNT], util.i2s)
    for v in vcf.aln_to_vars(refseqstr, predstr):
        pred_vars.append(v)

    for true_var in known_vars:
        true_var_type = util.var_type(true_var)
        if true_var in pred_vars:
            counts[true_var_type]['tp'] += 1
        else:
            counts[true_var_type]['fn'] += 1

    for detected_var in pred_vars:
        if detected_var not in known_vars:
            vartype = util.var_type(detected_var)
            counts[vartype]['fp'] += 1

    return counts



def calc_val_accuracy(loader, model, criterion):
    """
    Compute accuracy (fraction of predicted bases that match actual bases),
    calculates mean number of variant counts between tgt and predicted sequence,
    and also calculates TP, FP and FN count based on reference, alt and predicted sequence.
    across all samples in valpaths, using the given model, and return it
    :param valpaths: List of paths to (src, tgt) saved tensors
    :returns : Average model accuracy across all validation sets, vaf MSE 
    """
    model.eval()
    with torch.no_grad():
        match_sum0 = 0
        match_sum1 = 0
        result_totals0 = init_count_dict()
        result_totals1 = init_count_dict()

        var_counts_sum0 = 0
        var_counts_sum1 = 0
        tot_samples = 0
        total_batches = 0
        loss_tot = 0

        for src, tgt_kmers, vaf, *_ in loader.iter_once(64):
            total_batches += 1
            tot_samples += src.shape[0]
            seq_preds = util.predict_sequence(src, model, n_output_toks=37, device=DEVICE) # 150 // 4 = 37, this will need to be changed if we ever want to change the output length

            #tgt_kmers = util.tgt_to_kmers(tgt[:, :, 0:truncate_seq_len]).float().to(DEVICE)
            tgt_kmer_idx = torch.argmax(tgt_kmers, dim=-1)[:, :, 1:]
            seq_preds = seq_preds[:, :, 0:tgt_kmer_idx.shape[-1], :] # tgt_kmer_idx might be a bit shorter if the sequence is truncated
            if type(criterion) == nn.NLLLoss:
                loss_tot = compute_twohap_loss(seq_preds, tgt_kmer_idx, criterion)
            else:
                raise ValueError("Only cross entropy supported now")

            midmatch0, varcount0, results_totals0 = _calc_hap_accuracy(src, seq_preds[:, 0, :, :], tgt_kmer_idx[:, 0, :], result_totals0)
            midmatch1, varcount1, results_totals1 = _calc_hap_accuracy(src, seq_preds[:, 1, :, :], tgt_kmer_idx[:, 1, :], result_totals1)
            match_sum0 += midmatch0
            match_sum1 += midmatch1

            var_counts_sum0 += varcount0
            var_counts_sum1 += varcount1
                
    return (match_sum0 / total_batches,
            match_sum1 / total_batches,
            var_counts_sum0 / tot_samples,
            var_counts_sum1 / tot_samples,
            result_totals0, result_totals1,
            loss_tot)


def safe_compute_ppav(results0, results1, key):
    try:
        ppa = (results0[key]['tp'] + results1[key]['tp']) / (
                results0[key]['tp'] + results1[key]['tp'] + results0[key]['fn'] + results1[key]['fn'])
    except ZeroDivisionError:
        ppa = 0
    try:
        ppv = (results0[key]['tp'] + results1[key]['tp']) / (
                results0[key]['tp'] + results1[key]['tp'] + results0[key]['fp'] + results1[key]['fp'])
    except ZeroDivisionError:
        ppv = 0

    return ppa, ppv


def train_epochs(epochs,
                 dataloader,
                 max_read_depth=50,
                 feats_per_read=10,
                 init_learning_rate=0.0025,
                 checkpoint_freq=0,
                 statedict=None,
                 model_dest=None,
                 val_dir=None,
                 batch_size=64,
                 lossfunc='ce',
                 wandb_run_name=None,
                 wandb_notes="",
                 cl_args = {}
):
    encoder_attention_heads = 8 # was 4
    decoder_attention_heads = 4 # was 4
    dim_feedforward = 512
    encoder_layers = 4
    decoder_layers = 2 # was 2
    embed_dim_factor = 120 # was 100
    model = VarTransformer(read_depth=max_read_depth,
                            feature_count=feats_per_read, 
                            kmer_dim=util.FEATURE_DIM, # Number of possible kmers
                            n_encoder_layers=encoder_layers,
                            n_decoder_layers=decoder_layers,
                            embed_dim_factor=embed_dim_factor,
                            encoder_attention_heads=encoder_attention_heads,
                            decoder_attention_heads=decoder_attention_heads,
                            d_ff=dim_feedforward,
                            device=DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(MODEL_DTYPE).to(DEVICE)

    model_tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Creating model with {model_tot_params} trainable params")
    if statedict is not None:
        logger.info(f"Initializing model with state dict {statedict}")
        model.load_state_dict(torch.load(statedict, map_location=DEVICE))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    if lossfunc == 'ce':
        logger.info("Creating N.L.L. Loss loss function")
        criterion = nn.NLLLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.995)
    elif lossfunc == 'sw':
        gap_open_penalty = -5
        gap_exend_penalty = -1
        temperature = 1.0
        trim_width = 100
        logger.info(f"Creating Smith-Waterman loss function with gap open: {gap_open_penalty} extend: {gap_exend_penalty} temp: {temperature:.4f}, trim_width: {trim_width}")
        criterion = SmithWatermanLoss(gap_open_penalty=gap_open_penalty,
                                    gap_extend_penalty=gap_exend_penalty,
                                    temperature=temperature,
                                    trim_width=trim_width,
                                    device=DEVICE,
                                    reduction=None,
                                    window_mode="random")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.90)

    trainlogpath = str(model_dest).replace(".model", "").replace(".pt", "") + "_train.log"
    logger.info(f"Training log data will be saved at {trainlogpath}")


    trainlogger = TrainLogger(trainlogpath, [
            "epoch", "trainingloss", "val_accuracy",
            "mean_var_count", "ppa_dels", "ppa_ins", "ppa_snv",
            "ppv_dels", "ppv_ins", "ppv_snv", "learning_rate", "epochtime",
            "batch_time_mean", "decompress_frac", "train_frac", "io_frac",
            "zero_grad_frac", "forward_pass_frac", "loss_frac", "midmatch_frac",
            "backward_pass_frac", "optimize_frac",
    ])

    if ENABLE_WANDB:
        import wandb
        # get git branch info for logging
        git_repo = Repository(os.path.abspath(__file__))
        # what to log in wandb
        wandb_config_params = dict(
            learning_rate=init_learning_rate,
            embed_dim_factor=embed_dim_factor,
            feats_per_read=feats_per_read,
            batch_size=batch_size,
            read_depth=max_read_depth,
            encoder_attn_heads=encoder_attention_heads,
            decoder_attn_heads=decoder_attention_heads,
            transformer_dim=dim_feedforward,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            git_branch=git_repo.head.name,
            git_target=git_repo.head.target,
            model_param_count=model_tot_params,
            git_last_commit=next(git_repo.walk(git_repo.head.target)).message,
            loss_func=str(criterion),
        )
        # log command line too
        wandb_config_params.update(cl_args)

        # change working dir so wandb finds git repo info
        current_working_dir = os.getcwd()
        git_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(git_dir)

        wandb.init(
                config=wandb_config_params,
                project='variant-transformer',
                entity='arup-rnd',
                dir=current_working_dir,
                name=wandb_run_name,
                notes=wandb_notes,
                resume=True,
        )
        wandb.watch(model, log="all", log_freq=1000)

        # back to correct working dir
        os.chdir(current_working_dir)

    if val_dir:
        logger.info(f"Using validation data in {val_dir}")
        val_loader = loader.PregenLoader(device=DEVICE, datadir=val_dir, threads=4, tgt_prefix="tgkmers")
    else:
        logger.info(f"No val. dir. provided retaining a few training samples for validation")
        valpaths = dataloader.retain_val_samples(fraction=0.05)
        val_loader = loader.PregenLoader(device=DEVICE, datadir=None, pathpairs=valpaths, threads=4, tgt_prefix="tgkmers")
        logger.info(f"Pulled {len(valpaths)} samples to use for validation")

    try:
        for epoch in range(epochs):
            starttime = datetime.now()
            loss, epoch_times = train_epoch(model,
                                             optimizer,
                                             criterion,
                                             dataloader,
                                             batch_size=batch_size)

            elapsed = datetime.now() - starttime

            acc0, acc1, var_count0, var_count1, results0, results1, val_loss = calc_val_accuracy(val_loader, model, criterion)

            ppa_dels, ppv_dels = safe_compute_ppav(results0, results1, 'del')
            ppa_ins, ppv_ins = safe_compute_ppav(results0, results1, 'ins')
            ppa_snv, ppv_snv = safe_compute_ppav(results0, results1, 'snv')

            logger.info(f"Epoch {epoch} Secs: {elapsed.total_seconds():.2f} lr: {scheduler.get_last_lr()[0]:.4f} loss: {loss:.4f} val acc: {acc0:.3f} / {acc1:.3f}  ppa: {ppa_snv:.3f} / {ppa_ins:.3f} / {ppa_dels:.3f}  ppv: {ppv_snv:.3f} / {ppv_ins:.3f} / {ppv_dels:.3f}")


            if epoch_times.get("batch_count", 0) > 0:
                mean_batch_time = epoch_times.get("batch_time",0.0) / epoch_times.get("batch_count")
            else:
                mean_batch_time = 0.0
            if epoch_times.get("batch_time", 0.0) > 0.0:
                decompress_time_frac = epoch_times.get("decomp_time", 0.0) / epoch_times.get("batch_time")
                load_time_frac = epoch_times.get("load_time", 0.0) / epoch_times.get("batch_time")
                train_time_frac = epoch_times.get("train_time", 0.0) / epoch_times.get("batch_time")
                zero_grad_frac = epoch_times.get("zero_grad_time", 0.0) / epoch_times.get("batch_time")
                forward_pass_frac = epoch_times.get("forward_pass_time", 0.0) / epoch_times.get("batch_time")
                loss_frac = epoch_times.get("loss_time", 0.0) / epoch_times.get("batch_time")
                midmatch_frac = epoch_times.get("midmatch_time", 0.0) / epoch_times.get("batch_time")
                backward_pass_frac = epoch_times.get("backward_pass_time", 0.0) / epoch_times.get("batch_time")
                optimize_frac = epoch_times.get("optimize_time", 0.0) / epoch_times.get("batch_time")
            else:
                load_time_frac = 0.0
                decompress_time_frac = 0.0
                train_time_frac = 0.0
                zero_grad_frac = 0.0
                forward_pass_frac = 0.0
                loss_frac = 0.0
                midmatch_frac = 0.0
                backward_pass_frac = 0.0
                optimize_frac = 0.0

            if ENABLE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "trainingloss": loss,
                    "validation_loss": val_loss,
                    "accuracy/val_acc_hap0": acc0,
                    "accuracy/val_acc_hap1": acc1,
                    "accuracy/var_count0": var_count0,
                    "accuracy/var_count1": var_count1,
                    "accuracy/ppa dels": ppa_dels,
                    "accuracy/ppa ins": ppa_ins,
                    "accuracy/ppa snv": ppa_snv,
                    "accuracy/ppv dels": ppv_dels,
                    "accuracy/ppv ins": ppv_ins,
                    "accuracy/ppv snv": ppv_snv,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epochtime": elapsed.total_seconds(),
                    "batch_time_mean": mean_batch_time,
                    "performance/decompress_frac": decompress_time_frac,
                    "performance/train_frac": train_time_frac,
                    "performance/io_frac": load_time_frac,
                    "performance/zero_grad_frac": zero_grad_frac,
                    "performance/forward_pass_frac": forward_pass_frac,
                    "performance/loss_frac": loss_frac,
                    "performance/midmatch_frac": midmatch_frac,
                    "performance/backward_pass_frac": backward_pass_frac,
                    "performance/optimize_frac": optimize_frac,
                })

            scheduler.step()
            trainlogger.log({
                "epoch": epoch,
                "trainingloss": loss,
                "val_accuracy": acc0.item() if isinstance(acc0, torch.Tensor) else acc0,
                "mean_var_count": var_count0,
                "ppa_dels": ppa_dels,
                "ppa_ins": ppa_ins,
                "ppa_snv": ppa_snv,
                "ppv_dels": ppv_dels,
                "ppv_ins": ppv_ins,
                "ppv_snv": ppv_snv,
                "learning_rate": scheduler.get_last_lr()[0],
                "epochtime": elapsed.total_seconds(),
                "batch_time_mean": mean_batch_time,
                "decompress_frac": decompress_time_frac,
                "train_frac": train_time_frac,
                "io_frac": load_time_frac,
                "zero_grad_frac": zero_grad_frac,
                "forward_pass_frac": forward_pass_frac,
                "loss_frac": loss_frac,
                "midmatch_frac": midmatch_frac,
                "backward_pass_frac": backward_pass_frac,
                "optimize_frac": optimize_frac,
            })


            if epoch > -1 and checkpoint_freq > 0 and (epoch % checkpoint_freq == 0):
                modelparts = str(model_dest).rsplit(".", maxsplit=1)
                checkpoint_name = modelparts[0] + f"_epoch{epoch}." + modelparts[1]
                logger.info(f"Saving model state dict to {checkpoint_name}")
                m = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(m.state_dict(), checkpoint_name)

        logger.info(f"Training completed after {epoch} epochs")
    except KeyboardInterrupt:
        pass

    if model_dest is not None:
        logger.info(f"Saving model state dict to {model_dest}")
        m = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(m.to('cpu').state_dict(), model_dest)


def load_train_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    # assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


def init_count_dict():
    return {
        'del': defaultdict(int),
        'ins': defaultdict(int),
        'snv': defaultdict(int),
        'mnv': defaultdict(int),
    }



def train(config, output_model, input_model, epochs, **kwargs):
    """
    Conduct a training run and save the trained parameters (statedict) to output_model
    :param config: Path to config yaml
    :param output_model: Path to save trained params to
    :param input_model: Start training with params from input_model
    :param epochs: How many passes over training data to conduct
    """
    logger.info(f"Found torch device: {DEVICE}")
    if 'cuda' in str(DEVICE):
        for idev in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {idev} name: {torch.cuda.get_device_name({idev})}")

    logger.info(f"Using pregenerated training data from {kwargs.get('datadir')}")
    dataloader = loader.PregenLoader(DEVICE,
                                     kwargs.get("datadir"),
                                     threads=kwargs.get('threads'),
                                     max_decomped_batches=kwargs.get('max_decomp_batches'),
                                     tgt_prefix="tgkmers")


    # If you want to use augmenting loaders you need to pass '--data-augmentation" parameter during training, default is no augmentation.
    if kwargs.get("data_augmentation"):
        #dataloader = loader.ShorteningLoader(dataloader, seq_len=150)
        dataloader = loader.ShufflingLoader(dataloader)
        #dataloader = loader.DownsamplingLoader(dataloader, prob_of_read_being_dropped=0.01

    torch.cuda.empty_cache()   
    train_epochs(epochs,
                 dataloader,
                 max_read_depth=100,
                 feats_per_read=10,
                 statedict=input_model,
                 init_learning_rate=kwargs.get('learning_rate', 0.001),
                 model_dest=output_model,
                 checkpoint_freq=kwargs.get('checkpoint_freq', 10),
                 val_dir=kwargs.get('val_dir'),
                 batch_size=kwargs.get("batch_size"),
                 lossfunc=kwargs.get('loss'),
                 wandb_run_name=kwargs.get("wandb_run_name"),
                 wandb_notes=kwargs.get("wandb_notes"),
                 cl_args=kwargs.get("cl_args"),
                 )

