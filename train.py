
import logging
import yaml
from datetime import datetime

import torch
from torch import nn

logger = logging.getLogger(__name__)

import loader
import bwasim
from bam import string_to_tensor, target_string_to_tensor, encode_pileup3, reads_spanning, alnstart, ensure_dim
from model import VarTransformer, AltPredictor, VarTransformerAltMask


DEVICE = torch.device("cuda:0") if hasattr(torch, 'cuda') and torch.cuda.is_available() else torch.device("cpu")




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
    for unsorted_src, tgt_seq, tgtvaf, altmask in loader.iter_once(batch_size):
        # predicted_altmask = altpredictor(unsorted_src)
        # amx = 0.95 / predicted_altmask.max(dim=1)[0]
        # amin = predicted_altmask.min(dim=1)[0].unsqueeze(1).expand((-1, predicted_altmask.shape[1]))
        # predicted_altmask = (predicted_altmask - amin) * amx.unsqueeze(1).expand(
        #     (-1, predicted_altmask.shape[1])) + amin
        # predicted_altmask = predicted_altmask.clamp(0.001, 1.0)
        # predicted_altmask = torch.cat((torch.ones(unsorted_src.shape[0], 1).to(DEVICE), predicted_altmask[:, 1:]), dim=1)
        # aex = predicted_altmask.unsqueeze(-1).unsqueeze(-1)
        # fullmask = aex.expand(unsorted_src.shape[0], unsorted_src.shape[2], unsorted_src.shape[1], unsorted_src.shape[3]).transpose(1, 2)
        # src = unsorted_src * fullmask
        src = unsorted_src

        optimizer.zero_grad()
        
        seq_preds, vaf_preds = model(src)

        tgt_seq = tgt_seq.squeeze(1)
        loss = criterion(seq_preds.flatten(start_dim=0, end_dim=1), tgt_seq.flatten())

        count += 1
        if count % 100 == 0:
            logger.info(f"Batch {count} : epoch_loss_sum: {epoch_loss_sum:.3f}")
        #print(f"preds: {seq_preds.shape} tgt: {tgt_seq.shape}")
        # vafloss = vaf_criterion(vaf_preds.double().squeeze(1), tgtvaf.double())
        with torch.no_grad():
            width = 20
            mid = seq_preds.shape[1] // 2
            midmatch = (torch.argmax(seq_preds[:, mid-width//2:mid+width//2, :].flatten(start_dim=0, end_dim=1),
                                     dim=1) == tgt_seq[:, mid-width//2:mid+width//2].flatten()
                         ).float().mean()



        loss.backward(retain_graph=True)
        # vafloss.backward()
        optimizer.step()
        epoch_loss_sum += loss.detach().item()
        # vafloss_sum += vafloss.detach().item()

    return epoch_loss_sum, midmatch.item(), vafloss_sum


def train_epochs(epochs,
                 dataloader,
                 max_read_depth=50,
                 feats_per_read=8,
                 init_learning_rate=0.0025,
                 checkpoint_freq=0,
                 statedict=None,
                 model_dest=None,
                 eval_batches=None):

    model = VarTransformerAltMask(read_depth=max_read_depth, feature_count=feats_per_read, out_dim=4, nhead=6, d_hid=300, n_encoder_layers=2).to(DEVICE)
    logger.info(f"Creating model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")
    if statedict is not None:
        logger.info(f"Initializing model with state dict {statedict}")
        model.load_state_dict(torch.load(statedict))
    model.train()
    batch_size = 64

    # altpredictor = AltPredictor(0, 7)
    # altpredictor.load_state_dict(torch.load("altpredictor3.sd"))
    # altpredictor.to(DEVICE)
    
    altpredictor = None

    criterion = nn.CrossEntropyLoss()
    vaf_crit = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.995)
    try:
        for epoch in range(epochs):
            starttime = datetime.now()
            loss, refmatch, vafloss = train_epoch(model,
                                                  optimizer,
                                                  criterion,
                                                  vaf_crit,
                                                  dataloader,
                                                  batch_size=batch_size,
                                                  max_alt_reads=max_read_depth,
                                                  altpredictor=altpredictor)
            elapsed = datetime.now() - starttime

            logger.info(f"Epoch {epoch} Secs: {elapsed.total_seconds():.2f} lr: {scheduler.get_last_lr()[0]:.4f} loss: {loss:.4f} Ref match: {refmatch:.4f}  vafloss: {vafloss:.4f} ")
            scheduler.step()


            if epoch > 0 and checkpoint_freq > 0 and (epoch % checkpoint_freq == 0):
                modelparts = str(model_dest).rsplit(".", maxsplit=1)
                checkpoint_name = modelparts[0] + f"_epoch{epoch}" + modelparts[1]
                logger.info(f"Saving model state dict to {checkpoint_name}")
                torch.save(model.to('cpu').state_dict(), checkpoint_name)


            if eval_batches is not None:
                with torch.no_grad():
                    for vartype, (src, tgt, vaftgt, altmask) in eval_batches.items():
                        # Use 'altpredictor' to mask out non-alt reads
                        src = src.to(DEVICE)
                        predicted_altmask = altpredictor(src.to(DEVICE))
                        amx = 0.95 / predicted_altmask.max(dim=1)[0]
                        amin = predicted_altmask.min(dim=1)[0].unsqueeze(1).expand((-1, predicted_altmask.shape[1]))
                        predicted_altmask = (predicted_altmask - amin) * amx.unsqueeze(1).expand(
                            (-1, predicted_altmask.shape[1])) + amin
                        predicted_altmask = torch.cat((torch.ones(src.shape[0], 1).to(DEVICE), predicted_altmask[:, 1:]), dim=1)
                        predicted_altmask = predicted_altmask.clamp(0.001, 1.0)
                        aex = predicted_altmask.unsqueeze(-1).unsqueeze(-1)
                        fullmask = aex.expand(src.shape[0], src.shape[2], src.shape[1],
                                              src.shape[3]).transpose(1, 2).to(DEVICE)
                        src = src * fullmask

                        predictions, vafpreds = model(src.to(DEVICE))
                        tps, fps, fns = eval_batch(src, tgt, predictions)
                        #logger.info(f"Eval: Min alt mask: {minalt:.3f} max: {maxalt:.3f}")
                        if tps > 0:
                            logger.info(f"Eval: {vartype} PPA: {(tps / (tps + fns)):.3f} PPV: {(tps / (tps + fps)):.3f}")
                        else:
                            logger.info(f"Eval: {vartype} PPA: No TPs found :(")

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
    conf = load_train_conf(config)
    train_sets = [(c['bam'], c['labels']) for c in conf['data']]
    #dataloader = make_multiloader(train_sets, conf['reference'], threads=6, max_to_load=max_to_load, max_reads_per_aln=200)
    # dataloader = loader.SimLoader(DEVICE, seqlen=100, readsperbatch=100, readlength=80, error_rate=0.01, clip_prob=0.01)
    if kwargs.get("datadir") is not None:
        logger.info(f"Using pregenerated training data from {kwargs.get('datadir')}")
        dataloader = loader.PregenLoader(DEVICE, kwargs.get("datadir"))
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
                 model_dest=output_model,
                 eval_batches=eval_batches)

