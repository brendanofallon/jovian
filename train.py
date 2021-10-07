
import logging
import yaml
from datetime import datetime
import os

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

import loader
import bwasim
import util
from bam import string_to_tensor, target_string_to_tensor, encode_pileup3, reads_spanning, alnstart, ensure_dim
from model import VarTransformer, AltPredictor, VarTransformerAltMask, VarTransformerFixedAltMask

ENABLE_WANDB=False

if ENABLE_WANDB:
    import wandb
    wandb.init(project='variant-transformer', entity='arup-rnd')


DEVICE = torch.device("cuda:0") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")


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
        assert len(items) == len(self.headers), f"Expected {len(headers)} items to log, but got {len(items)}"
        self.output.write(
            ",".join(str(items[k]) for k in self.headers) + "\n"
        )
        self._flush_and_fsync()





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
    epoch_loss_sum = 0
    vafloss_sum = 0
    count = 0
    vafloss = torch.tensor([0])
    for batch, (src, tgt_seq, tgtvaf, altmask) in enumerate(loader.iter_once(batch_size)):
        optimizer.zero_grad()
        
        seq_preds, vaf_preds = model(src)

        tgt_seq = tgt_seq.squeeze(1)
        loss = criterion(seq_preds.flatten(start_dim=0, end_dim=1), tgt_seq.flatten())

        count += 1
        if count % 100 == 0:
            logger.info(f"Batch {count} : epoch_loss_sum: {epoch_loss_sum:.3f}")
        #print(f"src: {src.shape} preds: {seq_preds.shape} tgt: {tgt_seq.shape}")

        with torch.no_grad():
            width = 20
            mid = seq_preds.shape[1] // 2
            midmatch = (torch.argmax(seq_preds[:, mid-width//2:mid+width//2, :].flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt_seq[:, mid-width//2:mid+width//2].flatten()
                         ).float().mean()

        loss.backward(retain_graph=vaf_criterion is not None)

        #if vaf_criterion is not None and np.random.rand() < 0.10:
        #    vafloss = vaf_criterion(vaf_preds.double(), tgtvaf.double())
        #    vafloss.backward()
        #    vafloss_sum += vafloss.detach().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Not sure what is reasonable here, but we want to prevent the gradient from getting too big
        optimizer.step()
        epoch_loss_sum += loss.detach().item()
        if np.isnan(epoch_loss_sum):
            logger.warning(f"Loss is NAN!!")
        #logger.info(f"batch: {batch} loss: {loss.item()} vafloss: {vafloss.item()}")

    return epoch_loss_sum, midmatch.item(), vafloss_sum


def calc_val_accuracy(valpaths, model):
    """
    Compute accuracy (fraction of predicted bases that match actual bases)
    across all samples in valpaths, using the given model, and return it
    :param valpaths: List of paths to (src, tgt) saved tensors
    :returns : Average model accuracy across all validation sets, vaf MSE 
    """
    width = 20
    with torch.no_grad():
        match_sum = 0
        count = 0
        vaf_mse_sum = 0
        for srcpath, tgtpath, vaftgtpath in valpaths:
            src = util.tensor_from_file(srcpath, DEVICE).float()
            tgt = util.tensor_from_file(tgtpath, DEVICE).long()
            vaf = util.tensor_from_file(vaftgtpath, DEVICE)
            tgt = tgt.squeeze(1)

            seq_preds, vaf_preds = model(src)
            mid = seq_preds.shape[1] // 2
            midmatch = (torch.argmax(seq_preds[:, mid - width // 2:mid + width // 2, :].flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt[:, mid - width // 2:mid + width // 2].flatten()
                        ).float().mean()
            match_sum += midmatch
            count += 1

            vaf_mse_sum += ((vaf - vaf_preds) * (vaf-vaf_preds)).mean().item()
    return match_sum / count, vaf_mse_sum / count


def train_epochs(epochs,
                 dataloader,
                 max_read_depth=50,
                 feats_per_read=8,
                 init_learning_rate=0.0025,
                 checkpoint_freq=0,
                 statedict=None,
                 model_dest=None,
                 eval_batches=None,
                 val_dir=None):

    model = VarTransformerFixedAltMask(
        read_depth=max_read_depth,
        feature_count=feats_per_read,
        out_dim=4,
        nhead=6,
        d_hid=300,
        n_encoder_layers=2,
        altpredictor_sd="altpredictor_conv1x5_500.sd",
        device=DEVICE).to(DEVICE)
    logger.info(f"Creating model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
    if statedict is not None:
        logger.info(f"Initializing model with state dict {statedict}")
        model.load_state_dict(torch.load(statedict))
    model.train()
    batch_size = 64

    criterion = nn.CrossEntropyLoss()
    vaf_crit = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.995)

    trainlogpath = str(model_dest).replace(".model", "").replace(".pt", "") + "_train.log"
    logger.info(f"Training log data will be saved at {trainlogpath}")
    trainlogger = TrainLogger(trainlogpath, ["epoch", "trainingloss", "trainmatch", "valmatch", "learning_rate", "epochtime"])

    tensorboard_log_path = str(model_dest).replace(".model", "") + "_tensorboard_data"
    tensorboardWriter = SummaryWriter(log_dir=tensorboard_log_path)

    if ENABLE_WANDB:
        wandb.config.learning_rate = init_learning_rate
        wandb.config.batch_size = 64
        wandb.config.read_depth = max_read_depth
        wandb.watch(model)

    valpaths = []
    try:
        if val_dir:
            logger.info(f"Using validation data in {val_dir}")
            valpaths = load_train_data(val_dir)
        else:
            logger.info(f"No val. dir. provided retaining a few training samples for validation")
            valpaths = dataloader.retain_val_samples(fraction=0.05)
            logger.info(f"Pulled {len(valpaths)} samples to use for validation")

    except Exception as ex:
        logger.warning(f"Error loading validation samples, will not have any validation data for this run {ex}")
        valpaths = []
    
    try:
        for epoch in range(epochs):
            starttime = datetime.now()
            loss, train_accuracy, vafloss = train_epoch(model,
                                                  optimizer,
                                                  criterion,
                                                  vaf_crit,
                                                  dataloader,
                                                  batch_size=batch_size,
                                                  max_alt_reads=max_read_depth,
                                                  altpredictor=None)
            elapsed = datetime.now() - starttime

            if valpaths:
                val_accuracy, val_vaf_mse = calc_val_accuracy(valpaths, model)
            else:
                val_accuracy, val_vaf_mse = float("NaN"), float("NaN")
            logger.info(f"Epoch {epoch} Secs: {elapsed.total_seconds():.2f} lr: {scheduler.get_last_lr()[0]:.4f} loss: {loss:.4f} train acc: {train_accuracy:.4f} val accuracy: {val_accuracy:.4f}, val VAF accuracy: {val_vaf_mse:.4f}")

            if ENABLE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "trainingloss": loss,
                    "trainmatch": train_accuracy,
                    "valmatch": val_accuracy.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epochtime": elapsed.total_seconds(),
                })

            scheduler.step()
            trainlogger.log({
                "epoch": epoch,
                "trainingloss": loss,
                "trainmatch": train_accuracy,
                "valmatch": val_accuracy.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "epochtime": elapsed.total_seconds(),
            })

            tensorboardWriter.add_scalar("loss/train", loss, epoch)
            tensorboardWriter.add_scalar("match/train", train_accuracy, epoch)
            tensorboardWriter.add_scalar("match/val", val_accuracy, epoch)


            if epoch > 0 and checkpoint_freq > 0 and (epoch % checkpoint_freq == 0):
                modelparts = str(model_dest).rsplit(".", maxsplit=1)
                checkpoint_name = modelparts[0] + f"_epoch{epoch}." + modelparts[1]
                logger.info(f"Saving model state dict to {checkpoint_name}")
                torch.save(model.state_dict(), checkpoint_name)


        logger.info(f"Training completed after {epoch} epochs")
    except KeyboardInterrupt:
        pass

    if model_dest is not None:
        logger.info(f"Saving model state dict to {model_dest}")
        torch.save(model.to('cpu').state_dict(), model_dest)


def load_train_conf(confyaml):
    logger.info(f"Loading configuration from {confyaml}")
    conf = yaml.safe_load(open(confyaml).read())
    assert 'reference' in conf, "Expected 'reference' entry in training configuration"
    assert 'data' in conf, "Expected 'data' entry in training configuration"
    return conf


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


def create_eval_batches(batch_size, num_reads, read_length, config):
    """
    Create batches of simulated variants for evaluation
    :param batch_size: Number of pileups per batch
    :param config: Config yaml, must contain reference and regions
    :return: Mapping from variant type -> (batch src, tgt, vaftgt, altmask)
    """
    base_error_rate = 0.01
    if type(config) == str:
        conf = load_train_conf(config)
    else:
        conf = config
    regions = bwasim.load_regions(conf['regions'])
    eval_batches = {}
    logger.info(f"Generating evaluation batches of size {batch_size}")
    vaffunc = bwasim.betavaf
    eval_batches['del'] = bwasim.make_batch(batch_size,
                                                      regions,
                                                      conf['reference'],
                                                      numreads=num_reads,
                                                      readlength=read_length,
                                                      var_funcs=[bwasim.make_het_del],
                                                      vaf_func=vaffunc,
                                                      error_rate=base_error_rate,
                                                      clip_prob=0)
    eval_batches['ins'] = bwasim.make_batch(batch_size,
                                                regions,
                                                conf['reference'],
                                                numreads=num_reads,
                                                readlength=read_length,
                                                vaf_func=vaffunc,
                                                var_funcs=[bwasim.make_het_ins],
                                                error_rate=base_error_rate,
                                                clip_prob=0)
    eval_batches['snv'] = bwasim.make_batch(batch_size,
                                                regions,
                                                conf['reference'],
                                                numreads=num_reads,
                                                readlength=read_length,
                                                vaf_func=vaffunc,
                                                var_funcs=[bwasim.make_het_snv],
                                                error_rate=base_error_rate,
                                                clip_prob=0)
    eval_batches['mnv'] = bwasim.make_batch(batch_size,
                                                regions,
                                                conf['reference'],
                                                numreads=num_reads,
                                                readlength=read_length,
                                                vaf_func=vaffunc,
                                                var_funcs=[bwasim.make_het_del],
                                                error_rate=base_error_rate,
                                                clip_prob=0)
    logger.info("Done generating evaluation batches")
    return eval_batches


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
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(DEVICE)}")
    conf = load_train_conf(config)
    # train_sets = [(c['bam'], c['labels']) for c in conf['data']]
    #dataloader = make_multiloader(train_sets, conf['reference'], threads=6, max_to_load=max_to_load, max_reads_per_aln=200)
    # dataloader = loader.SimLoader(DEVICE, seqlen=100, readsperbatch=100, readlength=80, error_rate=0.01, clip_prob=0.01)
    if kwargs.get("datadir") is not None:
        logger.info(f"Using pregenerated training data from {kwargs.get('datadir')}")
        dataloader = loader.PregenLoader(DEVICE,
                                         kwargs.get("datadir"),
                                         threads=kwargs.get('threads'),
                                         max_decomped_batches=kwargs.get('max_decomp_batches'))
    else:
        logger.info(f"Using on-the-fly training data from sim loader")
        dataloader = loader.BWASimLoader(DEVICE,
                                     regions=conf['regions'],
                                     refpath=conf['reference'],
                                     readsperpileup=300,
                                     readlength=145,
                                     error_rate=0.02,
                                     clip_prob=0.01)
    eval_batches = None #create_eval_batches(25, 200, 145, conf)
    train_epochs(epochs,
                 dataloader,
                 max_read_depth=300,
                 feats_per_read=8,
                 statedict=input_model,
                 init_learning_rate=kwargs.get('learning_rate', 0.001),
                 model_dest=output_model,
                 checkpoint_freq=kwargs.get('checkpoint_freq', 10),
                 eval_batches=eval_batches)

