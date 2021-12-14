
import logging
import yaml
from datetime import datetime
import os
from pygit2 import Repository

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

import vcf
import loader
import util
from bam import string_to_tensor, target_string_to_tensor, encode_pileup3, reads_spanning, alnstart, ensure_dim
from model import VarTransformer
from swloss import SmithWatermanLoss

ENABLE_WANDB = os.getenv('ENABLE_WANDB', False)

if ENABLE_WANDB:
    import wandb


DEVICE = torch.device("cuda") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


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


def train_epoch(model, optimizer, criterion, vaf_criterion, loader, batch_size, max_alt_reads, altpredictor=None):
    """
    Train for one epoch, which is defined by the loader but usually involves one pass over all input samples
    :param model: Model to train
    :param optimizer: Optimizer to update params
    :param criterion: Loss function
    :param loader: Provides training data
    :param batch_size:
    :return: Sum of losses over each batch, plus fraction of matching bases for ref and alt seq
    """
    epoch_loss_sum = None
    prev_epoch_loss = None
    vafloss_sum = 0
    count = 0
    midmatch_hap0_sum = 0
    midmatch_hap1_sum = 0
    vafloss = torch.tensor([0])
    # init time usage to zero
    epoch_times = {}
    start_time = datetime.now()
    for batch, (src, tgt_seq, tgtvaf, altmask, log_info) in enumerate(loader.iter_once(batch_size)):
        if log_info:
            decomp_time = log_info.get("decomp_time", 0.0)
        else:
            decomp_time = 0.0
        times = dict(start=start_time, decomp_and_load=datetime.now(), decomp_time=decomp_time)

        optimizer.zero_grad()
        times["zero_grad"] = datetime.now()

        seq_preds = model(src)
        times["forward_pass"] = datetime.now()

        if type(criterion) == nn.CrossEntropyLoss:
            # Compute losses in both configurations, and use the best?
            #loss = criterion(seq_preds.flatten(start_dim=0, end_dim=2), tgt_seq.flatten())
            with torch.no_grad():
                for b in range(src.shape[0]):
                    loss1 = criterion(seq_preds[b, :, :, :].flatten(start_dim=0, end_dim=1), tgt_seq[b, :, :].flatten())
                    loss2 = criterion(seq_preds[b, :, :, :].flatten(start_dim=0, end_dim=1), tgt_seq[b, torch.tensor([1,0]), :].flatten())

                    if loss2 < loss1:
                        seq_preds[b, :, :, :] = seq_preds[b, torch.tensor([1,0]), :]

            loss = criterion(seq_preds.flatten(start_dim=0, end_dim=2), tgt_seq.flatten())
        else:
            with torch.no_grad():
                for b in range(src.shape[0]):
                    # Must look at two configurations, each with two losses...
                    loss1 = criterion(seq_preds[b, 0, :, :].unsqueeze(0), tgt_seq[b, 0, :].unsqueeze(0))
                    loss1 += criterion(seq_preds[b, 1, :, :].unsqueeze(0), tgt_seq[b, 1, :].unsqueeze(0))

                    loss2 = criterion(seq_preds[b, 0, :, :].unsqueeze(0), tgt_seq[b, 1, :].unsqueeze(0))
                    loss2 += criterion(seq_preds[b, 1, :, :].unsqueeze(0), tgt_seq[b, 0, :].unsqueeze(0))

                    if loss2 < loss1:
                        seq_preds[b, :, :, :] = seq_preds[b, torch.tensor([1, 0]), :]

            loss = criterion(seq_preds[:, 0, :, :], tgt_seq[:, 0, :])
            loss += criterion(seq_preds[:, 1, :, :], tgt_seq[:, 1, :])

        times["loss"] = datetime.now()

        count += 1
        if count % 100 == 0:
            if prev_epoch_loss:
                lossdif = epoch_loss_sum - prev_epoch_loss
            else:
                lossdif = 0
            logger.info(f"Batch {count} : epoch_loss_sum: {epoch_loss_sum:.3f} epoch loss dif: {lossdif:.3f}")
        #print(f"src: {src.shape} preds: {seq_preds.shape} tgt: {tgt_seq.shape}")

        with torch.no_grad():
            width = 200
            mid = seq_preds.shape[2] // 2
            midmatch_hap0 = (torch.argmax(seq_preds[:, 0, mid-width//2:mid+width//2, :].flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt_seq[:, 0, mid-width//2:mid+width//2].flatten()
                         ).float().mean()
            midmatch_hap0_sum += midmatch_hap0.item()

            midmatch_hap1 = (torch.argmax(seq_preds[:, 1, mid-width//2:mid+width//2, :].flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt_seq[:, 1, mid-width//2:mid+width//2].flatten()
                         ).float().mean()
            midmatch_hap1_sum += midmatch_hap1.item()

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
        if batch % 1 == 0:
            logger.info(f"Batch {batch} : hap0 / 1 acc: {midmatch_hap0.item():.3f} / {midmatch_hap1.item():.3f} hap1loss: {loss.item():.3f}")
            
        #logger.info(f"batch: {batch} loss: {loss.item()} vafloss: {vafloss.item()}")
        epoch_times = calc_time_sums(time_sums=epoch_times, **times)
        start_time = datetime.now()  # reset timer for next batch
        del src
    torch.cuda.empty_cache()
    logger.info(f"Trained {batch+1} batches in total.")
    return epoch_loss_sum, midmatch_hap0_sum / batch, midmatch_hap1_sum / batch, epoch_times


def _calc_hap_accuracy(src, seq_preds, tgt, width=100):
    # Compute val accuracy
    mid = seq_preds.shape[1] // 2
    midmatch = (torch.argmax(seq_preds[:, mid - width // 2:mid + width // 2, :].flatten(start_dim=0, end_dim=1),
                             dim=1) == tgt[:, mid - width // 2:mid + width // 2].flatten()
                ).float().mean()

    # Compute mean variant counts
    var_count = 0
    batch_size = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for b in range(src.shape[0]):
        predstr = util.readstr(seq_preds[b, :, :])
        tgtstr = util.tgt_str(tgt[b, :])
        var_count += len(list(vcf.aln_to_vars(tgtstr, predstr)))
        batch_size += 1

        # Get TP, FN and FN based on reference, alt and predicted sequence.
        tps, fps, fns = eval_prediction(util.readstr(src[b, :, 0, :]), tgtstr, seq_preds[b, :, :])
        tp_total += len(tps)
        fp_total += len(fps)
        fn_total += len(fns)

    return midmatch, var_count, tp_total, fp_total, fn_total


def calc_val_accuracy(loader, model, criterion):
    """
    Compute accuracy (fraction of predicted bases that match actual bases),
    calculates mean number of variant counts between tgt and predicted sequence,
    and also calculates TP, FP and FN count based on reference, alt and predicted sequence.
    across all samples in valpaths, using the given model, and return it
    :param valpaths: List of paths to (src, tgt) saved tensors
    :returns : Average model accuracy across all validation sets, vaf MSE 
    """
    width = 20
    with torch.no_grad():
        match_sum0 = 0
        match_sum1 = 0
        count = 0
        tp_tot0 = 0
        fp_tot0 = 0
        fn_tot0 = 0
        tp_tot1 = 0
        fp_tot1 = 0
        fn_tot1 = 0

        var_counts_sum0 = 0
        var_counts_sum1 = 0
        tot_samples = 0
        total_batches = 0
        for src, tgt, vaf, *_ in loader.iter_once(64):

            total_batches += 1
            tot_samples += src.shape[0]
            seq_preds = model(src)
            for b in range(src.shape[0]):
                loss1 = criterion(seq_preds[b, :, :].flatten(start_dim=0, end_dim=1), tgt[b, :, :].flatten())
                loss2 = criterion(seq_preds[b, :, :].flatten(start_dim=0, end_dim=1),
                                  tgt[b, torch.tensor([1, 0]), :].flatten())
                if loss2 < loss1:
                    seq_preds[b, :, :, :] = seq_preds[b, torch.tensor([1, 0]), :]

            midmatch0, varcount0, tps0, fps0, fns0 = _calc_hap_accuracy(src, seq_preds[:, 0, :, :], tgt[:, 0, :])
            midmatch1, varcount1, tps1, fps1, fns1 = _calc_hap_accuracy(src, seq_preds[:, 1, :, :], tgt[:, 1, :])
            match_sum0 += midmatch0
            tp_tot0 += tps0
            fp_tot0 += fps0
            fn_tot0 += fns0
            match_sum1 += midmatch1
            tp_tot1 += tps1
            fp_tot1 += fps1
            fn_tot1 += fns1

            var_counts_sum0 += varcount0
            var_counts_sum1 += varcount1
                
    return (match_sum0 / total_batches,
            match_sum1 / total_batches,
            var_counts_sum0 / tot_samples,
            var_counts_sum1 / tot_samples,
            tp_tot0, fp_tot0, fn_tot0,
            tp_tot1, fp_tot1, fn_tot1, )


def train_epochs(epochs,
                 dataloader,
                 max_read_depth=50,
                 feats_per_read=8,
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
    attention_heads = 2
    transformer_dim = 500
    encoder_layers = 6
    embed_dim_factor = 100
    model = VarTransformer(read_depth=max_read_depth,
                                    feature_count=feats_per_read, 
                                    out_dim=4,
                                    embed_dim_factor=embed_dim_factor,
                                    nhead=attention_heads, 
                                    d_hid=transformer_dim, 
                                    n_encoder_layers=encoder_layers,
                                    device=DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    logger.info(f"Creating model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
    if statedict is not None:
        logger.info(f"Initializing model with state dict {statedict}")
        model.load_state_dict(torch.load(statedict, map_location=DEVICE))
    model.train()

    if lossfunc == 'ce':
        logger.info("Creating CrossEntropy loss function")
        criterion = nn.CrossEntropyLoss()
    elif lossfunc == 'sw':
        gap_open_penalty=-5
        gap_exend_penalty=-1
        temperature=1.0
        trim_width=100
        logger.info(f"Creating Smith-Waterman loss function with gap open: {gap_open_penalty} extend: {gap_exend_penalty} temp: {temperature:.4f}, trim_width: {trim_width}")
        criterion = SmithWatermanLoss(gap_open_penalty=gap_open_penalty,
                                   gap_extend_penalty=gap_exend_penalty,
                                   temperature=temperature,
                                   trim_width=trim_width,
                                   device=DEVICE)

    vaf_crit = None #nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.995)

    trainlogpath = str(model_dest).replace(".model", "").replace(".pt", "") + "_train.log"
    logger.info(f"Training log data will be saved at {trainlogpath}")


    trainlogger = TrainLogger(trainlogpath, [
            "epoch",
            "trainingloss",
            "train_accuracy",
            "val_accuracy",
            "mean_var_count",
            "ppa0",
            "ppv0",
            "learning_rate",
            "epochtime",
            "batch_time_mean",
            "decompress_frac",
            "train_frac",
            "io_frac",
            "zero_grad_frac",
            "forward_pass_frac",
            "loss_frac",
            "midmatch_frac",
            "backward_pass_frac",
            "optimize_frac",
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
            attn_heads=attention_heads,
            transformer_dim=transformer_dim,
            encoder_layers=encoder_layers,
            git_branch=git_repo.head.name,
            git_target=git_repo.head.target,
            git_last_commit=next(git_repo.walk(git_repo.head.target)).message,
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
                name=wandb_run_name,
                notes=wandb_notes,
        )
        wandb.watch(model, log="all", log_freq=1000)

        # back to correct working dir
        os.chdir(current_working_dir)

    if val_dir:
        logger.info(f"Using validation data in {val_dir}")
        val_loader = loader.PregenLoader(device=DEVICE, datadir=val_dir, threads=4)
    else:
        logger.info(f"No val. dir. provided retaining a few training samples for validation")
        valpaths = dataloader.retain_val_samples(fraction=0.05)
        val_loader = loader.PregenLoader(device=DEVICE, datadir=None, pathpairs=valpaths, threads=4)
        logger.info(f"Pulled {len(valpaths)} samples to use for validation")

    try:
        for epoch in range(epochs):
            starttime = datetime.now()
            loss, train_acc0, train_acc1, epoch_times = train_epoch(model,
                                                                 optimizer,
                                                                 criterion,
                                                                 vaf_crit,
                                                                 dataloader,
                                                                 batch_size=batch_size,
                                                                 max_alt_reads=max_read_depth)

            elapsed = datetime.now() - starttime

            acc0, acc1, var_count0, var_count1, tps0, fps0, fns0, tps1, fps1, fns1 = calc_val_accuracy(val_loader, model, criterion)

            try:
                ppa0 = tps0/(tps0+fns0)
                ppv0 = tps0/(tps0+fps0)
                ppa1 = tps1/(tps1+fns1)
                ppv1 = tps1/(tps1+fps1)
            except ZeroDivisionError:
                ppa0 = 0
                ppv0 = 0
                ppa1 = 0
                ppv1 = 0

            logger.info(f"Epoch {epoch} Secs: {elapsed.total_seconds():.2f} lr: {scheduler.get_last_lr()[0]:.4f} loss: {loss:.4f} train acc: {train_acc0:.4f} / {train_acc1:.4f}, val acc: {acc0:.3f} / {acc1:.3f} var counts: {var_count0:.3f} / {var_count1:.3f}, ppa: {ppa0:.3f} / {ppa1:.3f}, ppv: {ppv0:.3f} / {ppv1:.3f}")


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
                    "train_acc_hap0": train_acc0,
                    "train_acc_hap1": train_acc1,
                    "accuracy/val_acc_hap0": acc0,
                    "accuracy/val_acc_hap1": acc1,
                    "accuracy/var_count0": var_count0,
                    "accuracy/var_count1": var_count1,
                    "accuracy/ppa hap0": ppa0,
                    "accuracy/ppv hap0": ppv0,
                    "accuracy/ppa hap1": ppa1,
                    "accuracy/ppv hap1": ppv1,
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
                "train_accuracy": train_acc0,
                "val_accuracy": acc0,
                "mean_var_count": var_count0,
                "ppa0": ppa0,
                "ppv0": ppv0,
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


            if epoch > 0 and checkpoint_freq > 0 and (epoch % checkpoint_freq == 0):
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
    assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


def eval_prediction(refseqstr, altseq, predictions, midwidth=100):
    """
    Given a target sequence and two predicted sequences, attempt to determine if the correct *Variants* are
    detected from the target. This uses the vcf.align_seqs(seq1, seq2) method to convert string sequences to Variant
    objects, then compares variant objects
    :param tgt:
    :param predictions:
    :param midwidth:
    :return: Sets of TP, FP, and FN vars
    """
    midstart = len(refseqstr) // 2 - midwidth // 2
    midend  =  len(refseqstr) // 2 + midwidth // 2
    known_vars = []
    for v in vcf.aln_to_vars(refseqstr, altseq):
        if midstart < v.pos < midend:
            known_vars.append(v)

    pred_vars = []
    predstr = util.readstr(predictions)
    for v in vcf.aln_to_vars(refseqstr, predstr):
        if midstart < v.pos < midend:
            v.qual = predictions[v.pos:v.pos + max(1, min(len(v.ref), len(v.alt))), :].max(dim=1)[0].min().item()
            pred_vars.append(v)

    tps = [] # True postive - real and detected variant
    fns = [] # False negatives - real variant but not detected
    fps = [] # False positives - detected but not a real variant
    for true_var in known_vars:
        if true_var in pred_vars:
            tps.append(true_var)
        else:
            fns.append(true_var)

    for detected_var in pred_vars:
        if detected_var not in known_vars:
            #logger.info(f"FP: {detected_var}")
            fps.append(detected_var)

    return tps, fps, fns


def eval_batch(src, tgt, predictions):
    """
    Run evaluation on a single batch and report number of TPs, FPs, and FNs
    :param src: Model input (with batch dimension as first dim and ref sequence as first element in dimension 2)
    :param tgt: Model targets / true alt sequence
    :param predictions: Model prediction
    :return: Total number of TP, FP, and FN variants
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for b in range(src.shape[0]):
        refseq = src[b, :, 0, :]
        assert refseq[:, 0:4].sum() == refseq.shape[0], f"Probable incorrect refseq index, sum did not match sequence length!"
        tps, fps, fns = eval_prediction(refseq, tgt[b, :], predictions[b, :, :])
        tp_total += len(tps)
        fp_total += len(fps)
        fn_total += len(fns)
    return tp_total, fp_total, fn_total


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
 
    conf = load_train_conf(config)

    if kwargs.get("datadir") is not None:
        logger.info(f"Using pregenerated training data from {kwargs.get('datadir')}")
        pregenloader = loader.PregenLoader(DEVICE,
                                         kwargs.get("datadir"),
                                         threads=kwargs.get('threads'),
                                         max_decomped_batches=kwargs.get('max_decomp_batches'))
        
        dataloader = pregenloader

        # If you want to use augmenting loaders you need to pass '--data-augmentation" parameter during training, default is no augmentation.
        if kwargs.get("data_augmentation"):
            dataloader = loader.ShorteningLoader(dataloader, seq_len=150)
            dataloader = loader.ShufflingLoader(dataloader)
            dataloader = loader.DownsamplingLoader(dataloader, prob_of_read_being_dropped=0.01)

    else:
        logger.info(f"Using on-the-fly training data from sim loader")
        dataloader = loader.BWASimLoader(DEVICE,
                                     regions=conf['regions'],
                                     refpath=conf['reference'],
                                     readsperpileup=300,
                                     readlength=145,
                                     error_rate=0.02,
                                     clip_prob=0.01)
    torch.cuda.empty_cache()   
    train_epochs(epochs,
                 dataloader,
                 max_read_depth=100,
                 feats_per_read=9,
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

