import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class Feedforward(nn.Module):

    def __init__(self,input_dim,output_dim,mid_dim,use_check_point=False,norm_type='L'):
        super(Feedforward,self).__init__()
        self.input_dim=input_dim
        self.mid_dim=mid_dim
        self.output_dim=output_dim
        self.norm_type=norm_type
        self.use_check_point=use_check_point

        self.ffd=nn.Sequential(
            nn.Linear(self.input_dim,self.mid_dim),
            nn.GELU(),
            nn.Linear(self.mid_dim,self.output_dim)
        )

        if self.norm_type=='L':
            self.norm=nn.LayerNorm(self.output_dim)


    def forward(self,inputs):
        if self.use_check_point:
            ans=cp.checkpoint(self.check_forward,inputs,use_reentrant=True)
        else:
            ans=self.check_forward(inputs)
        return ans

    def check_forward(self,inputs):
        ans=self.norm(self.ffd(inputs)+inputs)
        return ans

