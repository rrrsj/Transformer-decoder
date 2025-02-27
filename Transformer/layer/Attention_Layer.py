import torch
import torch.nn as nn
import os
import torch.utils.checkpoint as cp


class Attention_Layer(nn.Module):
    def __init__(self,input_dim,num_heads,embeding_dim,output_dim,use_check_point=False,use_group_attention=False,group_num=4,norm_type='L'):
        super(Attention_Layer,self).__init__()

        assert num_heads%group_num==0
        self.input_dim=input_dim
        self.use_group_attention=use_group_attention
        self.num_heads=num_heads
        self.group_num=group_num
        self.embeding_dim=embeding_dim
        self.output_dim=output_dim
        self.use_check_point=use_check_point
        self.kv_heads=num_heads//group_num


        self.embeding_Q=nn.Sequential(
            nn.Linear(self.input_dim,self.embeding_dim*self.num_heads)
        )
        
        self.embeding_K=nn.Sequential(
            nn.Linear(self.input_dim,self.embeding_dim*self.kv_heads)
        )

        self.embeding_V=nn.Sequential(
            nn.Linear(self.input_dim,self.embeding_dim*self.kv_heads)
        )

        self.output_linear=nn.Sequential(
            nn.Linear(self.num_heads*self.embeding_dim,self.output_dim),
            nn.GELU()
        )

        if norm_type=='L':
            self.norm=nn.LayerNorm(self.output_dim)

    def forward(self,inputs):
        batch_size=inputs.size()[0]
        length=inputs.size()[1]
        if self.use_check_point:
            ans=cp.checkpoint(self.check_forward,inputs,use_reentrant=True)
        else:
            ans=self.check_forward(inputs)
        
        ans=self.norm(inputs+self.output_linear(ans))
        return ans
        

    def get_decoder_mask(self,input_length):
        decoder_mask=torch.triu(torch.ones(input_length,input_length),diagonal=1).to(os.environ.get('device'))
        return decoder_mask

    def check_forward(self,inputs):
        batch_size=inputs.size()[0]
        length=inputs.size()[1]
        Q=self.embeding_Q(inputs).reshape(batch_size,length,self.num_heads,self.embeding_dim).transpose(1,2)
        K=self.embeding_K(inputs).reshape(batch_size,length,1,self.kv_heads,self.embeding_dim).expand(-1,-1,self.group_num,-1,-1).reshape(batch_size,length,self.num_heads,self.embeding_dim).transpose(1,2)
        V=self.embeding_V(inputs).reshape(batch_size,length,1,self.kv_heads,self.embeding_dim).expand(-1,-1,self.group_num,-1,-1).reshape(batch_size,length,self.num_heads,self.embeding_dim).transpose(1,2)
        attention_value=torch.einsum("bnle,bnef->bnlf",Q,K.transpose(2,3))/(self.embeding_dim**0.5)
        decoder_mask=self.get_decoder_mask(length).reshape(1,1,length,length).expand(batch_size,self.num_heads,-1,-1)
        attention_score=torch.softmax(attention_value+(decoder_mask*(-1e9)),dim=-1)
        ans=torch.einsum('bnlf,bnfe->bnle',attention_score,V).transpose(1,2).reshape(batch_size,length,-1)
        return ans