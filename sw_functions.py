import jax
import jax.numpy as jnp
import os
import torch

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Written by Sergey Ovchinnikov and Sam Petti 
# Spring 2021

def sw_nogap(batch=True, unroll=2):
  '''smith-waterman (local alignment) with no gap'''
  # rotate matrix for striped dynamic-programming

  def sw_rotate(x, mask=None):
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    zero = jnp.zeros([n,m])
    if mask is None: mask = 1.0
    output = {"x":zero.at[i,j].set(x),
              "m":zero.at[i,j].set(mask),
              "o":(jnp.arange(n)+a%2)%2}
    prev = (jnp.zeros(m), jnp.zeros(m))
    return output,prev,(i,j)

  # comute scoring (hij) matrix
  def sw_sco(x, lengths, temp=1.0):
    def _soft_maximum(x, axis=None):
      return temp*jax.nn.logsumexp(x/temp,axis)
    def _cond(cond, true, false):
      return cond*true + (1-cond)*false

    def _step(prev, sm):      
      h2,h1 = prev   # previous two rows of scoring (hij) mtx
      h1_T = _cond(sm["o"],jnp.pad(h1[:-1],[1,0]),jnp.pad(h1[1:],[0,1]))
      h0 = jnp.stack([h2+sm["x"], h1, h1_T],-1)
      h0 = sm["m"] * _soft_maximum(h0,-1)
      return (h1,h0),h0

    # make mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:,None] * (jnp.arange(b) < real_b)[None,:]

    sm,prev,idx = sw_rotate(x,mask=mask)
    hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
    return hij.max()

  # traceback (aka backprop) to get alignment
  traceback = jax.grad(sw_sco)

  # add batch dimension
  if batch: return jax.vmap(traceback,(0,0,None))
  else: return traceback

def sw(unroll=2, batch=True, NINF=-1e30):
  '''smith-waterman (local alignment) with gap parameter'''

  # rotate matrix for striped dynamic-programming
  def rotate(x):   
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    output = {"x":jnp.full([n,m],NINF).at[i,j].set(x), "o":(jnp.arange(n)+a%2)%2}
    return output, (jnp.full(m, NINF), jnp.full(m, NINF)), (i,j)

  # compute scoring (hij) matrix
  def sco(x, lengths, gap=0, temp=1.0):

    def _soft_maximum(x, axis=None, mask=None):
      def _logsumexp(y):
        y = jnp.maximum(y,NINF)
        if mask is None: return jax.nn.logsumexp(y, axis=axis)
        else: return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))
      return temp*_logsumexp(x/temp)
    
    def _cond(cond, true, false): return cond*true + (1-cond)*false
    def _pad(x,shape): return jnp.pad(x,shape,constant_values=(NINF,NINF))

    def _step(prev, sm):      
      h2,h1 = prev   # previous two rows of scoring (hij) mtx
      h1_T = _cond(sm["o"],_pad(h1[:-1],[1,0]),_pad(h1[1:],[0,1]))
      
      # directions
      Align = h2 + sm["x"]
      Turn_0 = h1 + gap
      Turn_1 = h1_T + gap
      Sky = sm["x"]

      h0 = jnp.stack([Align, Turn_0, Turn_1, Sky], -1)
      h0 = _soft_maximum(h0, -1)
      return (h1,h0),h0

    # mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:,None] * (jnp.arange(b) < real_b)[None,:]
    x = x + NINF * (1 - mask)

    sm, prev, idx = rotate(x[:-1,:-1])
    hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
    return _soft_maximum(hij + x[1:,1:], mask=mask[1:,1:])

  # traceback (aka backprop) to get alignment
  traceback = jax.grad(sco)

  # add batch dimension
  if batch: return jax.vmap(traceback,(0,0,None,None))
  else: return traceback


def sw_affine(restrict_turns=True, 
             penalize_turns=True,
             batch=True, unroll=2, NINF=-1e30):
  """smith-waterman (local alignment) with affine gap"""
  # rotate matrix for vectorized dynamic-programming
  

  def rotate(x):   
    # solution from jake vanderplas (thanks!)
    a, b = x.shape
    ar, br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i, j = (br-ar)+(a-1), (ar+br)//2
    n, m = (a+b-1), (a+b)//2
    output = {
      "x":jnp.full([n,m],NINF).at[i,j].set(x),
      "o":(jnp.arange(n)+a%2)%2
    }
    return output, (jnp.full((m,3), NINF), jnp.full((m,3), NINF)), (i,j)

  # fill the scoring matrix
  def sco(x, lengths, gap=0.0, open=0.0, temp=1.0):

    def _soft_maximum(x, axis=None, mask=None):

      def _logsumexp(y):
        y = jnp.maximum(y,NINF)
        if mask is None:
          return jax.nn.logsumexp(y, axis=axis)
        else:
          return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))

      return temp * _logsumexp(x/temp)

    def _cond(cond, true, false):
      return cond*true + (1-cond)*false

    def _pad(x,shape):
      return jnp.pad(x, shape, constant_values=(NINF, NINF))
      
    def _step(prev, sm): 
      h2,h1 = prev   # previous two rows of scoring (hij) mtxs
      print(f"Inside: dim of sm[x]: {sm['x'].shape} sm[o]: {sm['o'].shape}")
      p = jnp.pad(h2, [[0, 0], [0, 1]])
      print(f"Dim padded: {p.shape}")
      Align = jnp.pad(h2,[[0,0],[0,1]]) + sm["x"][:,None]
      Right = _cond(sm["o"], _pad(h1[:-1], ([1,0],[0,0])), h1)
      Down  = _cond(sm["o"], h1, _pad(h1[1:], ([0,1],[0,0])))

      # add gap penalty
      if penalize_turns:
        Right += jnp.stack([open, gap, open])
        Down += jnp.stack([open, open, gap])
      else:
        gap_pen = jnp.stack([open, gap, gap])
        Right += gap_pen
        Down += gap_pen

      if restrict_turns:
        Right = Right[:,:2]
      
      h0_Align = _soft_maximum(Align, -1)
      h0_Right = _soft_maximum(Right, -1)
      h0_Down = _soft_maximum(Down, -1)
      h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
      return (h1,h0), h0

    # mask
    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]
    x = x + NINF * (1 - mask)

    sm, prev, idx = rotate(x[:-1, :-1])
    print(f"Dim of sm[x]: {sm['x'].shape} sm[o]: {sm['o'].shape}")
    raw = jax.lax.scan(_step, prev, sm, unroll=unroll)
    hij = raw[-1][idx]
    # sink
    return _soft_maximum(hij + x[1:,1:,None], mask=mask[1:,1:,None])

  # traceback to get alignment (aka. get marginals)
  traceback = jax.grad(sco)

  # add batch dimension
  if batch:
    return jax.vmap(traceback,(0,0,None,None,None))
  else:
    return traceback


def rotate_jax(x, NINF=-1e30):
  # solution from jake vanderplas (thanks!)
  a, b = x.shape
  ar, br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
  i, j = (br-ar)+(a-1), (ar+br)//2
  n, m = (a+b-1), (a+b)//2
  output = {
    "x": jnp.full([n,m],NINF).at[i,j].set(x),
    "o": (jnp.arange(n)+a%2)%2
  }
  return output, (jnp.full((m,3), NINF), jnp.full((m,3), NINF)), (i,j)

def rotate_pyt(x, neg_inf=-1e30):
  ar = torch.arange(x.shape[0]-1, end=-1, step=-1).unsqueeze(1)
  br = torch.arange(x.shape[1]).unsqueeze(0)

  i = (br - ar) + (x.shape[0] - 1)
  j = (ar + br) // 2
  n = x.shape[0] + x.shape[1] - 1
  m = (x.shape[0] + x.shape[1]) // 2
  y = torch.full((n, m), neg_inf)
  y[i,j] = x
  output = {
    "x": y,
    "o": (torch.arange(n) + x.shape[0] % 2) % 2,
  }
  return output, (torch.full((m,3), neg_inf), torch.full((m,3), neg_inf)), (i, j)


def _soft_maximum(x, temp, axis=None, mask=None):
  NINF=-1e30
  def _logsumexp(y):
    y = jnp.maximum(y, NINF)
    if mask is None:
      return jax.nn.logsumexp(y, axis=axis)
    else:
      return y.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(y - y.max(axis, keepdims=True)), axis=axis))

  return temp * _logsumexp(x / temp)

def softmax_temperature_pyt(x, temperature):
  assert len(x.shape) == 2, "Need 2-dim input"
  return temperature * torch.logsumexp(x/temperature, dim=(0,1))


def _cond(cond, true, false):
  return cond * true + (1 - cond) * false

def _step(prev, sm, neg_inf=-1e30, penalize_turns=False):
  h2, h1 = prev  # previous two rows of scoring (hij) mtxs

  restrict_turns = False
  open = -5
  gap = -1

  Align = jnp.pad(h2, [[0, 0], [0, 1]]) + sm["x"][:, None]
  Right = _cond(sm["o"], jnp.pad(h1[:-1], ([1, 0], [0, 0]), constant_values=(neg_inf, neg_inf)), h1)
  Down =  _cond(sm["o"], h1, jnp.pad(h1[1:], ([0, 1], [0, 0]), constant_values=(neg_inf, neg_inf)))

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

  h0_Align = _soft_maximum(Align, temp=1, axis=-1)
  h0_Right = _soft_maximum(Right, temp=1, axis=-1)
  h0_Down = _soft_maximum(Down, temp=1, axis=-1)
  h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
  return (h1, h0), h0


def cond(cond, true, false):
  return cond * true + (1 - cond) * false

def step_pyt(prev, sm, neg_inf=-1e30):
  h2, h1 = prev  # previous two rows of scoring (hij) mtxs

  align = torch.nn.functional.pad(h2, ([0, 0], [0, 1])) + sm["x"].unsqueeze(0)
  right = cond(sm["o"], torch.nn.functional.pad(h1[:-1], ([1, 0], [0, 0]), value=neg_inf), h1)
  down = cond(sm["o"], h1, torch.nn.functional.pad(h1[1:], ([0, 1], [0, 0]), value=neg_inf))


def test():
  import numpy as np
  inputmat = np.array([[0.1, 0.2, 0.3, 0.4],
                       [0.11, 0.21, 0.31, 0.41],
                       [0.12, 0.22, 0.32, 0.42],
                       [0.13, 0.23, 0.33, 0.43]])

  # inputmat = jnp.array(np.random.random((10,10)))
  sw_func = sw_affine(batch=False)
  full_result = sw_func(jnp.array(inputmat), (inputmat.shape[0], inputmat.shape[1]))


  out_jx, prev_jx, ij_jx = rotate_jax(inputmat[:-1, :-1])
  carry = prev_jx
  y = out_jx
  results = []
  for i in range(out_jx['x'].shape[0]):
    carry, y = _step(carry, {
                    'x': out_jx['x'][i, :],
                    'o': out_jx['o'][i:i+1,],
              })
    results.append(y)

  hij = results[-1][ij_jx]
  final = _soft_maximum(hij + inputmat[1:, 1:, None], temp=1.0)

  out_pt, prev_pt, ij_pt = rotate_pyt(torch.tensor(inputmat[:-1, :-1]).float())

  print(out_jx["x"].copy())
  print(out_pt["x"].numpy())
  assert np.allclose(out_jx["x"].copy(), out_pt["x"].numpy())
  assert np.allclose(out_jx["o"].copy(), out_pt["o"].numpy())

  assert np.allclose(prev_jx[0].copy(), prev_pt[0].numpy())
  assert np.allclose(prev_jx[1].copy(), prev_pt[1].numpy())

  assert np.allclose(ij_jx[0].copy(), ij_pt[0].numpy())
  assert np.allclose(ij_jx[1].copy(), ij_pt[1].numpy())
  # assert out_jx["o"].numpy() == out_pt["o"].numpy()


  # jax_smt = _soft_maximum(inputmat, temp=17.0)

  result_jax = _step(prev_jx, out_jx)
  result = step_pyt(prev_pt, out_pt)


if __name__=="__main__":
  test()