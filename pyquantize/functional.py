import quantize
import torch

def quantizeNF4(weight:torch.Tensor,block_size:int=64):
    absmax,quant = quantize.quantizeNF4(weight,block_size)
    quant_state = [absmax,weight.shape]
    return quant,quant_state
    