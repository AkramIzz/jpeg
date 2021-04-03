#%%
from math import pi
from PIL import Image
import numpy as np

def image_to_numpy(image_path: str) -> np.ndarray:
  return np.array(Image.open(image_path))
#%%
def rgb_to_yCbCr(img: np.ndarray) -> np.ndarray:
  out = img.copy()
  out[:, :, 0] = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
  out[:, :, 1] = -0.169 * img[:, :, 0] - 0.331 * img[:, :, 1] + 0.5 * img[:, :, 2] + 128
  out[:, :, 2] = 0.5 * img[:, :, 0] - 0.419 * img[:, :, 1] - 0.081 * img[:, :, 2] + 128
  return out
#%%
def chroma_subsampling(channel: np.ndarray) -> np.ndarray:
  ''' 4:2:0 chroma subsampling '''
  assert len(channel.shape) == 2 and \
    channel.shape[0] % 2 == 0 and channel.shape[1] % 2 == 0
  out = channel.copy()
  out[1::2, :] = out[::2, :]
  out[:, 1::2] = out[:, ::2]
  return out
#%%
# TODO use matrix multiplication
def dct2d(inp: np.ndarray) -> np.ndarray:
  assert len(inp.shape) == 2 and inp.shape[0] == inp.shape[1]
  out = np.zeros(inp.shape)
  for u in range(inp.shape[0]):
    for v in range(inp.shape[0]):
      sum = 0
      for x in range(inp.shape[0]):
        for y in range(inp.shape[0]):
          sum += inp[x][y] * np.cos(((2*x+1)*u*pi)/16) * np.cos(((2*y+1)*v*pi)/16)
      if u == 0: Cu = 1/np.sqrt(2)
      else: Cu = 1
      if v == 0: Cv = 1/np.sqrt(2)
      else: Cv = 1
      out[u][v] = 1/4 *Cu*Cv*sum
  return out
#%%
def entropy_encode(channel: np.ndarray):
  # TODO
  pass
#%%
def store(img):
  # TODO
  pass
# %%
