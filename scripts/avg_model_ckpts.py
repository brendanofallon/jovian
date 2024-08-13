#!/usr/bin/env python

import sys
import torch
import torch.nn as nn
from collections import OrderedDict

torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)


sys.path.append("/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/dnaseq2seq")

from model import VarTransformer

# 100M conf
modelconf={
    "encoder_attention_heads": 8,  # was 4
    "decoder_attention_heads" : 10,  # was 4
    "dim_feedforward" : 512,
    "encoder_layers" : 10,
    "decoder_layers" : 10,  # was 2
    "embed_dim_factor": 160 , # was 100
    "max_read_depth": 150,
    "feats_per_read": 10,
}

def load_model(ckpt, modelconf=modelconf):
    # 100M params
    statedict = None
    if ckpt is not None:
        if 'model' in ckpt:
            statedict = ckpt['model']
            new_state_dict = {}
            for key in statedict.keys():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = statedict[key]
            statedict = new_state_dict
        else:
            statedict = ckpt

        if 'conf' in ckpt:
            print(f"Found model conf AND a checkpoint with model conf - using the model params from checkpoint")
            modelconf = ckpt['conf']

    model = VarTransformer(read_depth=modelconf['max_read_depth'],
                           feature_count=modelconf['feats_per_read'],
                           kmer_dim=260,  # Number of possible kmers
                           n_encoder_layers=modelconf['encoder_layers'],
                           n_decoder_layers=modelconf['decoder_layers'],
                           embed_dim_factor=modelconf['embed_dim_factor'],
                           encoder_attention_heads=modelconf['encoder_attention_heads'],
                           decoder_attention_heads=modelconf['decoder_attention_heads'],
                           d_ff=modelconf['dim_feedforward'],
                           device='cpu')

    if statedict is not None:
        print(f"Initializing model weights from state dict")
        model.load_state_dict(statedict)
    
    return model


def average_model_parameters(model_list):
    """
    Averages the parameters of the given list of models.
    All models must have the same architecture.

      :param model_list: List of models (instances of nn.Module)
      :return: A new model instance (nn.Module) with averaged parameters
    """
    # Start by getting a copy of the first model's state_dict
    # which will be used to store the average parameters
    average_state_dict = OrderedDict(model_list[0].state_dict())
    for key in average_state_dict:
        if key == "decoder0.layers.3.linear1.bias":
            print(f"{key}: {average_state_dict[key][0:20]}")

    # Initialize the averaged state dict with zeros
    #for key in average_state_dict:
    #    average_state_dict[key].zero_()

    # Sum all model parameters
    for model in model_list[1:]:
        model_state_dict = model.state_dict()
        for key in average_state_dict:
            if key == "decoder0.layers.3.linear1.bias":
                print(f"{key}: {model_state_dict[key][0:20]}")
            average_state_dict[key] += model_state_dict[key]

    # Divide by the number of models to get the average
    print(f"Final avg params:")
    for key in average_state_dict:
        average_state_dict[key] /= len(model_list)
        if key == "decoder0.layers.3.linear1.bias":
            print(f"{key}: {average_state_dict[key][0:20]}")


    new_model = load_model(average_state_dict)
    return new_model


def main(paths):
    models = []
    for path in paths:
        models.append(load_model(torch.load(path, map_location='cpu')))
    avg_model = average_model_parameters(models)

    final = {
        "model": avg_model.state_dict(),
        "conf": modelconf,
        "model_average_from": [paths]
    }
    torch.save(final, "averaged_model.pt")

if __name__=="__main__":
    main(sys.argv[1:])

