import torch
import torch.nn as nn
class Output_Layer(nn.Module):
    def __init__(self,input_dim,mid_dim,output_dim):
        super(Output_Layer,self).__init__()
        self.input_dim=input_dim
        self.mid_dim=mid_dim
        self.output_dim=output_dim

        self.output_linear=nn.Sequential(
            nn.Linear(self.input_dim,self.mid_dim),
            nn.GELU(),
            nn.Linear(self.mid_dim,self.output_dim)
        )

    def forward(self,inputs):
        
        ans=self.output_linear(inputs)
        
        return torch.softmax(ans,dim=-1) 