import pytest
import torch
import quantize
from bitsandbytes import functional as F

def test_quantizeNF4():
    for _ in range(10):
        inputs = torch.rand(4096,4096,dtype=torch.bfloat16,device="cuda")
        block_size = 64
        absmax,out = quantize.quantizeNF4(inputs,block_size)
        a,b = F.quantize_4bit(A=inputs,blocksize=block_size,quant_type="nf4")
        assert out.allclose(a)
        assert absmax.allclose(b[0])