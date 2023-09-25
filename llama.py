from typing import Tuple
import torch

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, hidden_size:int,eps:float=1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_eps = eps
    
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        variance = torch.mean(x*x,dim=-1,keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_eps)
        return self.weight * x
    
    def reset_parameter(self)->None:
        torch.nn.init.ones_(self.weight)
        
def precompute_freqs(dim:int,max_seq_length:int,base:float=10_000.0):
    """
    Precomputing frequency that are need for the rotary positional info.
    cos(m theta) and sin(m theta) 
    m --> max seq len
    num of theta will be multi head dim by 2 ie (hd/2)
    
    Refer section 3.4.2 in https://arxiv.org/pdf/2104.09864.pdf
    """
    freqs = 1.0 / base ** (torch.arange(0,dim,2).float()/dim)                   # shape [hd/2]
    t = torch.arange(max_seq_length,device=freqs.device)                        # shape [ms]
    freqs = torch.einsum("s,d->sd",t,freqs)                                     # shape [ms,hd/2]
    freqs = torch.polar(torch.ones_like(freqs),freqs)                           # shape [ms,hd/2] each element in complex form (cos mt + j sin mt)
    return freqs

def apply_rotary_emb(
    xq:torch.Tensor,                                                            # shape [b s nh hd]
    xk:torch.Tensor,                                                            # shape [b s nh hd]
freqs:torch.Tensor                                                              # shape [s hd/2] complex form
)->Tuple[torch.Tensor,torch.Tensor]:
    # we need convert the xq and xk into complex form
    xq_  = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1]),-1,2)       # shape [b s nh hd] --> [b s nh hd/2 2] --> [b s nh hd/2] complex
    xk_  = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1]),-1,2)       # shape [b s nh hd] --> [b s nh hd/2 2] --> [b s nh hd/2] complex
    # change the shape freq for broadcast
    freqs = (freqs.unsqueeze(0)                                                 # insert batch dim
                  .unsqueeze(2)                                                 # insert num of head, dim
                  )                                                             # shape [1 s 1 hd/2]
    # element wise multiplication in complex form then convert to real form
    xq_ = (torch.view_as_real(
                torch.einsum("bshd,bshd->bshd",xq_,freqs)                       # shape [b s nh hd/2] complex
                )
                .flatten(3)                                                     # shape [b s nh hd/2 2] real 
            )                                                                   # shape [b s nh hd] real         
    xk_ = (torch.view_as_real(
                torch.einsum("bshd,bshd->bshd",xk_,freqs)                       # shape [b s nh hd/2] complex
                )
                .flatten(3)                                                     # shape [b s nh hd/2 2] real 
            )                                                                   # shape [b s nh hd] real         
    return xq_.type_as(xq),xk_.type_as(xq)