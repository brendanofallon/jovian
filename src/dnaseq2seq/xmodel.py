from typing import Optional, Callable, Union, Tuple

from torch import Tensor
from torch.nn import Module
from x_transformers import TransformerWrapper, AutoregressiveWrapper, Encoder, Decoder
import torch
import torch.nn as nn
from x_transformers.autoregressive_wrapper import top_k
import torch.nn.functional as F

class SimpleDecodeWrapper(AutoregressiveWrapper):

    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)

    def generate(
        self,
        x,
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Optional[Tensor] = None,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Optional[Union[Module, Tuple[Module]]] = None,
        filter_kwargs: dict = dict(),
        cache_kv = True,
        **kwargs
    ):
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        b, t = x.shape[0], x.shape[1]

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None

        # output from which sampled tokens appended to

        out = torch.zeros((b, 1))
        # kv caches

        cache = None

        # sampling up to seq_len

        for _ in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (
                            cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embeeding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if cache is not None:
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits, new_cache = self.net(
                x,
                return_intermediates=True,
                cache=cache,
                seq_start_pos=seq_start_pos,
                **kwargs
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # filter by top_k, top_p (nucleus), top_a, or custom

            if greedy:
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            # concat sample

            out = torch.cat((out, sample), dim=-1)


        out = out[:, t:]

        return out


def main():
    x = torch.rand(17, 25, 260)


    tw = TransformerWrapper(
        num_tokens=
    )