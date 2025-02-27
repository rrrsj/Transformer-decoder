import torch
import torch.nn as nn
import math
class Position_Embeding(nn.Module):
    def __init__(self,input_dim,length):
        super(Position_Embeding,self).__init__()

        self.input_dim=input_dim
        self.length=length
        
        self.embeding_learn=nn.Parameter(torch.zeros(1,self.length,self.input_dim))
        self.embeding_static=torch.zeros(1,self.length,self.input_dim)

        for i in range(self.length):
            for j in range(self.input_dim):
                if j%2==0:
                    self.embeding_static[0][i][j]=math.sin(i/(10000**(j/self.input_dim)))
                else:
                    self.embeding_static[0][i][j]=math.sin(i/(10000**((j-1)/self.input_dim)))
        
    def forward(self,inputs):
        batch_size=inputs.size()[0]
        
        return inputs+self.embeding_learn.expand(batch_size,-1,-1)+self.embeding_static.expand(batch_size,-1,-1)