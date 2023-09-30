import pytest
import torch
from pyquantize import functional as py_func
from bitsandbytes import functional as bnb_func

def test_quantizeNF4():
    for _ in range(10):
        inputs = torch.rand(4096,4096,dtype=torch.bfloat16,device="cuda")
        block_size = 64
        out,quant_state = py_func.quantizeNF4(inputs,block_size)
        a,b = bnb_func.quantize_4bit(A=inputs,blocksize=block_size,quant_type="nf4")
        assert out.allclose(a)
        assert quant_state[0].allclose(b[0])