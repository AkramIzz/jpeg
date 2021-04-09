#%%
from math import pi
from PIL import Image
import numpy as np
#%%
def image_to_numpy(image_path: str) -> np.ndarray:
  return np.array(Image.open(image_path))
#%%
_rgb_yuv_conv = np.array([
  #     R,      G,      B   factors for
  [ 0.299,  0.587,  0.114], # Y 
  [-0.169, -0.331,    0.5], # Cb
  [   0.5, -0.419, -0.081], # Cr
]).T
def rgb_to_yCbCr(img: np.ndarray) -> np.ndarray:
  img = img.astype(np.float)
  out = img.dot(_rgb_yuv_conv)
  out[:, :, (1, 2)] += 128
  return out.astype(np.uint8)

_yuv_rgb_conv = np.array([
  #Y,    Cb,      Cr
  [1,      0,  1.402], # R
  [1, -0.334, -0.714], # G
  [1,  1.772,      0], # B
]).T
def yCbCr_to_rgb(yuv: np.ndarray) -> np.ndarray:
  yuv = yuv.astype(np.float)
  yuv[:, :, (1, 2)] -= 128
  out = yuv.dot(_yuv_rgb_conv)
  return out.astype(np.uint8)
#%%
def chroma_subsampling(yuv: np.ndarray) -> np.ndarray:
  ''' 4:2:0 chroma subsampling '''
  out = yuv.copy()
  out[1::2, :, (1, 2)] = out[::2, :, (1, 2)]
  out[:, 1::2, (1, 2)] = out[:, ::2, (1, 2)]
  return out
#%%
def dct1d_n(inp: np.ndarray) -> np.ndarray:
  N = inp.shape[-2]
  k = np.arange(N)
  n = np.arange(N)
  t = pi/N * (n+0.5)
  return inp @ np.cos(np.outer(k, t)).T

def inv_dct1d_n(inp: np.ndarray) -> np.ndarray:
  N = inp.shape[-2]
  k = np.arange(N)
  n = np.arange(1, N)
  t = pi/N * n
  # The various transpose operations and the bit hacky `[0:1]` instead of `[0]`
  # is used to support 2D arrays. The operation still operates on a 1D basis,
  # i.e. for a 2D array the operation performs the DCT-III on each row 
  # individually.
  dims = len(inp.shape)
  dims_perm = [i for i in range(dims)]
  tr = lambda x: np.transpose(x, (*dims_perm[:-2], dims-1, dims-2))
  dct3 = (0.5*tr(tr(inp)[..., 0:1, :])) + np.dot(tr(tr(inp)[..., 1:, :]), np.cos(np.outer(k+0.5, t)).T)
  # inverse of DCT-II is a scaled DCT-III
  return 2/N * dct3

def dct2d_n(inp: np.ndarray, inverse=False) -> np.ndarray:
  '''
  Apply 2D DCT-II or 2D inverse DCT-II on `inp`

    inp: multidimensional array. (inverse) DCT is applied to last two dimensions
  '''
  dims = len(inp.shape)
  dims_perm = [i for i in range(dims)]
  tr = lambda x: np.transpose(x, (*dims_perm[:-2], dims-1, dims-2))
  if inverse:
    return tr(inv_dct1d_n(tr(inv_dct1d_n(inp))))
  return tr(dct1d_n(tr(dct1d_n(inp))))
#%%
def dct1d(inp: np.ndarray) -> np.ndarray:
  assert len(inp.shape) in [1, 2]
  N = inp.shape[0]
  k = np.arange(N)
  n = np.arange(N)
  t = pi/N * (n+0.5)
  return np.dot(inp, np.cos(np.outer(k, t)).T)

def inv_dct1d(inp: np.ndarray) -> np.ndarray:
  N = inp.shape[0]
  k = np.arange(N)
  n = np.arange(1, N)
  t = pi/N * n
  # The various transpose operations and the bit hacky `[0:1]` instead of `[0]`
  # is used to support 2D arrays. The operation still operates on a 1D basis,
  # i.e. for a 2D array the operation performs the DCT-III on each row 
  # individually.
  dct3 = (0.5*inp.T[0:1].T) + np.dot(inp.T[1:].T, np.cos(np.outer(k+0.5, t)).T)
  # inverse of DCT-II is a scaled DCT-III
  return 2/N * dct3

def dct2d(inp: np.ndarray, inverse=False) -> np.ndarray:
  ''' Apply 2D DCT-II or 2D inverse DCT-II on `inp` '''
  assert len(inp.shape) == 2
  if inverse:
    return np.round(inv_dct1d(inv_dct1d(inp).T).T, decimals=10)
  return np.round(dct1d(dct1d(inp).T).T, decimals=10)
#%%
precomputed_quantization_table = np.array([
  [ 16,  12,  10,  16,  24,  40,  51,  61],
  [ 11,  12,  14,  19,  26,  58,  60,  55],
  [ 14,  13,  16,  24,  40,  57,  69,  56],
  [ 14,  17,  22,  29,  51,  87,  80,  62],
  [ 18,  22,  37,  56,  68, 109, 103,  77],
  [ 24,  35,  55,  64,  81, 104, 113,  92],
  [ 49,  64,  78,  87, 103, 121, 120, 103],
  [ 72,  92,  95,  98, 112, 100, 101,  99]
])

def gen_quantization_table(quality: int) -> np.ndarray:
  Q = np.empty((8, 8))
  for i in range(8):
      for j in range(8):
          Q[i, j] = 1 + (1+i+j) * quality
  return Q

def quantize(inp: np.ndarray, q: np.ndarray) -> np.ndarray:
  return np.round(inp / q)

def dequantize(inp: np.ndarray, q: np.ndarray):
  return inp * q
#%%
def entropy_encode(channel: np.ndarray):
  # TODO
  pass

def dpcm(inp: np.ndarray) -> np.ndarray:
  out = inp.copy()
  for i in range(inp.shape[0]-1):
    dc = inp[i, ..., 0, 0]
    out[i+1, ..., 0, 0] -= dc
  return out

def un_dpcm(inp: np.ndarray) -> np.ndarray:
  out = inp.copy()
  for i in range(inp.shape[0]-1):
    dc = out[i, ..., 0, 0]
    out[i+1, ..., 0, 0] += dc
  return out

_zigzag = (
  (0, 0),
  (0, 1), (1, 0),
  (2, 0), (1, 1), (0, 2),
  (0, 3), (1, 2), (2, 1), (3, 0),
  (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
  (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0),
  (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
  (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0),
  (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
  (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2),
  (7, 3), (6, 4), (5, 5), (4, 6), (3, 7),
  (4, 7), (5, 6), (6, 5), (7, 4),
  (7, 5), (6, 6), (5, 7),
  (6, 7), (7, 6),
  (7, 7),
)
_zigzag_rows = tuple(map(lambda x: x[0], _zigzag))
_zigzag_cols = tuple(map(lambda x: x[1], _zigzag))
def zigzag(inp: np.ndarray) -> np.ndarray:
  return inp[..., _zigzag_rows, _zigzag_cols]

_unzigzag = (
   0,  1,  5,  6, 14, 15, 27, 28,
   2,  4,  7, 13, 16, 26, 29, 42,
   3,  8, 12, 17, 25, 30, 41, 43,
   9, 11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63,
)
def unzigzag(inp: np.ndarray, shape=None) -> np.ndarray:
  return inp[..., _unzigzag].reshape(shape)
#%%
def store(img):
  # TODO
  pass
# %%
def blocks_split(inp: np.ndarray, block_dim=8) -> np.ndarray:
  row_blocks_indicies = [i for i in range(block_dim, inp.shape[0], block_dim)]
  col_blocks_indicies = [i for i in range(block_dim, inp.shape[1], block_dim)]
  out = np.array(list(map(
    lambda a: np.array_split(a, col_blocks_indicies, axis=1),
    np.array_split(inp, row_blocks_indicies)
  )))
  out = out.reshape(-1, *out.shape[-3:])
  # blocks, channels, width, height
  return np.moveaxis(out, 3, 1)
# %%
def de_blocks(inp: np.ndarray, width: int, height: int) -> np.ndarray:
  B = width // 8
  rows = [np.concatenate(inp[i*B:(i+1)*B], axis=2) for i in range(B)]
  img = np.concatenate(rows, axis=1)
  # width, height, channels
  return np.moveaxis(img, 0, 2)
