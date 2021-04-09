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
def de_blocks(inp: np.ndarray) -> np.ndarray:
  rows = [np.concatenate(inp[i*64:(i+1)*64], axis=2) for i in range(64)]
  img = np.concatenate(rows, axis=1)
  return np.moveaxis(img, 0, 2)
