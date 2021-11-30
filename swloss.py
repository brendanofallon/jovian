
import torch
import torch.nn as nn


class SmithWatermanLoss(nn.Module):
    """
    A pure pytorch implementation of differentiable Smith-Waterman alignment, following the
    algorithm first developed by Petti et al. 2021 (http://repository.cshl.edu/id/eprint/40409/)
    See also: https://github.com/spetti/SMURF
    """

    def __init__(
        self,
        gap_open_penalty,
        gap_extend_penalty,
        temperature=1.0,
        penalize_turns=True,
        restrict_turns=True,
        trim_width=None,
        device='cpu',
    ):
        super().__init__()
        self.device = device
        self.gap_open_penalty = gap_open_penalty
        self.gap_extend_penalty = gap_extend_penalty
        self.penalize_turns = penalize_turns
        self.restrict_turns = restrict_turns
        self.temperature = temperature
        self.trim_width = trim_width
        if self.penalize_turns:
            self.rightmod = torch.tensor(
                [self.gap_open_penalty, self.gap_extend_penalty, self.gap_open_penalty],
                device=self.device
            )
            self.downmod = torch.tensor(
                [self.gap_open_penalty, self.gap_open_penalty, self.gap_extend_penalty],
                device=self.device
            )
        else:
            self.rightmod = torch.tensor(
                [self.gap_open_penalty, self.gap_extend_penalty, self.gap_extend_penalty],
                device=self.device
            )
            self.downmod = torch.tensor(
                [self.gap_open_penalty, self.gap_open_penalty, self.gap_extend_penalty],
                device=self.device
            )

        # This is what is used in the original implementation, but we should probably make it data type-dependent
        self.neg_inf = -1e30

    def _gen_rotation_mat(self, shape):
        ar = torch.arange(shape[1] - 1, end=-1, step=-1, device=self.device).unsqueeze(1)
        br = torch.arange(shape[2], device=self.device).unsqueeze(0)

        i = (br - ar) + (shape[1] - 1)
        j = (ar + br) // 2
        n = shape[1] + shape[2] - 1
        m = (shape[1] + shape[2]) // 2
        y = torch.full((shape[0], n, m), self.neg_inf, device=self.device)
        return (
            y,
            (torch.arange(n, device=self.device) + shape[1] % 2) % 2,
            (torch.full((shape[0], m, 3), self.neg_inf, device=self.device), torch.full((shape[0], m, 3), self.neg_inf, device=self.device)),
            (i, j),
        )

    def _rotate(self, x):
        """
        Rotate the input matrix... by 45 degrees!
        """
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
        """ A softmax-with-temperature implementation """
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

    def forward(self, predictions, targets):
        """
        Expect predictions to have dimension [batch, seq, probs], and
        targets to have dimension [batch, seq, label]
        """
        if self.trim_width is not None:
            start = predictions.shape[1] // 2 - self.trim_width // 2
            end = predictions.shape[1] // 2 + self.trim_width // 2
            predictions = predictions[:, start:end, :]
            targets = targets[:, start:end]

        targs_onehot = nn.functional.one_hot(targets, num_classes=4).float().to(self.device)
        x = torch.bmm(predictions, targs_onehot.transpose(1, 2))
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
        return -1.0 * final.sum()



def example():
    """
    Some example code for using the Smith-Waterman loss...
    """
    BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # These functions just convert strings into the tensors needed for input
    def seq2tensor(s):
        t = torch.zeros(len(s), 4)
        for i, b in enumerate(s):
            t[i, BASE_IDX[b]] = 1
        return t

    def seq2labels(s):
        return torch.tensor([BASE_IDX[b] for b in s], dtype=torch.int)

    swl = SmithWatermanLoss(gap_open_penalty=-2, gap_extend_penalty=-1, temperature=1.0, device='cpu')
    s1 = seq2tensor("ACTGACTG")
    t1 = seq2labels("ACTGCCTG")

    loss = swl(s1.unsqueeze(0), t1.unsqueeze(0).long())
    print(f"Score: {loss.item()}")
    # loss.backward() # Backprop those gradients!


if __name__=="__main__":
    example()
