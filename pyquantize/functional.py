import quantize
import torch

def quantizeNF4(weight:torch.Tensor,block_size:int=64):
    absmax,quant = quantize.quantizeNF4(weight,block_size)
    quant_state = [absmax,weight.shape]
    return quant,quant_state
    
def dequantizeNF4(quant:torch.Tensor,quant_state:list[torch.Tensor,torch.Size]):
    weight = quantize.dequantizeNF4(quant,quant_state[0],quant_state[1])
    return weight