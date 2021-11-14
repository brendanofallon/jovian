import jax
import jax.numpy as jnp
import os
import torch

from jax.config import config
import torch.nn as nn

# config.update('jax_disable_jit', True)


os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Written by Sergey Ovchinnikov and Sam Petti
# Spring 2021


def sw_affine(
    restrict_turns=True, penalize_turns=True, batch=True, unroll=2, NINF=-1e30
):
    """smith-waterman (local alignment) with affine gap"""

    # rotate matrix for vectorized dynamic-programming

    def rotate(x):
        # solution from jake vanderplas (thanks!)
        a, b = x.shape
        ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2
        output = {
            "x": jnp.full([n, m], NINF).at[i, j].set(x),
            "o": (jnp.arange(n) + a % 2) % 2,
        }
        return output, (jnp.full((m, 3), NINF), jnp.full((m, 3), NINF)), (i, j)

    # fill the scoring matrix
    def sco(x, lengths, gap=0.0, open=0.0, temp=1.0):
        def _soft_maximum(x, axis=None, mask=None):
            def _logsumexp(y):
                y = jnp.maximum(y, NINF)
                if mask is None:
                    return jax.nn.logsumexp(y, axis=axis)
                else:
                    return y.max(axis) + jnp.log(
                        jnp.sum(
                            mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis
                        )
                    )

            return temp * _logsumexp(x / temp)

        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _pad(x, shape):
            return jnp.pad(x, shape, constant_values=(NINF, NINF))

        def _step(prev, sm):
            h2, h1 = prev  # previous two rows of scoring (hij) mtxs

            Align = jnp.pad(h2, [[0, 0], [0, 1]]) + sm["x"][:, None]
            Right = _cond(sm["o"], _pad(h1[:-1], ([1, 0], [0, 0])), h1)
            Down = _cond(sm["o"], h1, _pad(h1[1:], ([0, 1], [0, 0])))

            # add gap penalty
            if penalize_turns:
                Right += jnp.stack([open, gap, open])
                Down += jnp.stack([open, open, gap])
            else:
                gap_pen = jnp.stack([open, gap, gap])
                Right += gap_pen
                Down += gap_pen

            if restrict_turns:
                Right = Right[:, :2]

            h0_Align = _soft_maximum(Align, -1)
            h0_Right = _soft_maximum(Right, -1)
            h0_Down = _soft_maximum(Down, -1)
            h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
            return (h1, h0), h0

        # mask
        a, b = x.shape
        real_a, real_b = lengths
        mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]
        x = x + NINF * (1 - mask)

        sm, prev, idx = rotate(x[:-1, :-1])
        hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]

        # sink
        return _soft_maximum(hij + x[1:, 1:, None], mask=mask[1:, 1:, None])

    # traceback to get alignment (aka. get marginals)
    traceback = jax.grad(sco)

    # add batch dimension
    if batch:
        return jax.vmap(traceback, (0, 0, None, None, None))
    else:
        return traceback



class SmithWatermanLoss(nn.Module):
    def __init__(
        self,
        gap_open_penalty,
        gap_extend_penalty,
        temperature=1.0,
        penalize_turns=True,
        restrict_turns=True,
    ):
        super().__init__()
        self.gap_open_penalty = gap_open_penalty
        self.gap_extend_penalty = gap_extend_penalty
        self.penalize_turns = penalize_turns
        self.restrict_turns = restrict_turns
        self.temperature = temperature
        if self.penalize_turns:
            self.rightmod = torch.tensor(
                [self.gap_open_penalty, self.gap_extend_penalty, self.gap_open_penalty]
            )
            self.downmod = torch.tensor(
                [self.gap_open_penalty, self.gap_open_penalty, self.gap_extend_penalty]
            )
        else:
            self.rightmod = torch.tensor(
                [self.gap_open_penalty, self.gap_extend_penalty, self.gap_extend_penalty]
            )
            self.downmod = torch.tensor(
                [self.gap_open_penalty, self.gap_open_penalty, self.gap_extend_penalty]
            )

        self.neg_inf = -1e30

    def _gen_rotation_mat(self, shape):
        ar = torch.arange(shape[1] - 1, end=-1, step=-1).unsqueeze(1)
        br = torch.arange(shape[2]).unsqueeze(0)

        i = (br - ar) + (shape[1] - 1)
        j = (ar + br) // 2
        n = shape[1] + shape[2] - 1
        m = (shape[1] + shape[2]) // 2
        y = torch.full((shape[0], n, m), self.neg_inf)
        return (
            y,
            (torch.arange(n) + shape[1] % 2) % 2,
            (torch.full((shape[0], m, 3), self.neg_inf), torch.full((shape[0], m, 3), self.neg_inf)),
            (i, j),
        )

    def _rotate(self, x):
        smx, smo, carry, (i,j) = self._gen_rotation_mat(x.shape)
        smx[:, i, j] = x
        return (
            (smx, smo),
            carry,
            (i,j),
        )

    def _cond(self, cond, true, false):
        return cond * true + (1 - cond) * false

    def softmax_temperature(self, x, dim=-1):
        return self.temperature * torch.logsumexp(x / self.temperature, dim=dim)

    def _step(self, prev, smx, smo):
        h2, h1 = prev  # previous two rows of scoring (hij) mtxs

        align = torch.nn.functional.pad(h2, (0, 1, 0, 0)) + smx.unsqueeze(-1)
        right = self._cond(
            smo, torch.nn.functional.pad(h1[:,:-1,:], (0, 0, 1, 0), value=self.neg_inf), h1
        ) + self.rightmod
        down = self._cond(
            smo, h1, torch.nn.functional.pad(h1[:,1:,:], (0, 0, 0, 1), value=self.neg_inf)
        ) + self.downmod


        if self.restrict_turns:
            right = right[:, :, :2]

        h0_align = self.softmax_temperature(align)
        h0_right = self.softmax_temperature(right)
        h0_down = self.softmax_temperature(down)
        h0 = torch.stack([h0_align, h0_right, h0_down], dim=-1)
        return (h1, h0), h0

    def forward(self, x):
        (smx, smo), carry, (i,j) = self._rotate(x[:, :-1, :-1])
        results = []
        for n in range(smx.shape[1]):
            carry, y = self._step(
                carry,
                smx[:, n, :],
                smo[n,],
            )
            results.append(y)
        results = torch.stack(results).transpose(0,1)
        hij = results[:, i, j]
        final = self.softmax_temperature(hij + x[:, 1:, 1:, None], dim=(1, 2, 3))
        return final.sum()


def test():
    import numpy as np

    inputmat = np.random.random((5, 17, 23))

    sw_func = sw_affine(batch=True)
    full_result = sw_func(
        jnp.array(inputmat), jnp.array([[inputmat.shape[1], inputmat.shape[2]] for _ in range(inputmat.shape[0])]), -1, -5, 1.0
    )

    t = torch.tensor(inputmat, dtype=torch.float, requires_grad=True)
    swloss = SmithWatermanLossBatch(
        gap_open_penalty=-5, gap_extend_penalty=-1, temperature=1.0
    )
    pytresult = swloss(t)
    pytresult.backward()
    assert np.allclose(full_result.copy(), t.grad.detach().numpy())
    print(f"pyt: {pytresult}")

if __name__ == "__main__":
    test()
