#%%
from utils import *
import numpy as np

def encode(img_file: str) -> np.ndarray:
  img = image_to_numpy(img_file)
  yuv = rgb_to_yCbCr(img)
  yuv = chroma_subsampling(yuv)
  blocks = blocks_split(yuv)
  coeffs = dct2d_n(blocks)
  q = quantize(coeffs, precomputed_quantization_table)
  dc_encoded = dpcm(q)
  zz = zigzag(dc_encoded)
  return zz

def decode(inp: np.ndarray, width: int, height: int) -> np.ndarray:
  rdc_encoded = unzigzag(inp)
  q = un_dpcm(rdc_encoded)
  rcoeffs = dequantize(q, precomputed_quantization_table)
  rblocks = dct2d_n(rcoeffs, inverse=True)
  ryuv = de_blocks(rblocks, width, height)
  rimg = yCbCr_to_rgb(ryuv)
  return rimg
