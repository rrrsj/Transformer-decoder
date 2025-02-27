import torch 
import torch.nn as nn
from layer.Attention_Layer import Attention_Layer
from layer.Embeding import Embeding
from layer.Feedforward import Feedforward
from layer.Output_Layer import Output_Layer
from layer.Position_Embeding import Position_Embeding
from modelscope import AutoTokenizer
import transformers
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

class Decoder_Only(nn.Module):
    def __init__(self,num_layer,vocab_size,num_header,input_length,embeding_dim,use_group_attention=False,group_num=1,use_check_point=False):
        super(Decoder_Only,self).__init__()

        assert ((not use_group_attention) and (group_num==1))==False

        self.model_name="./OpenBMB/MiniCPM3-4B"
        self.vocab_size=vocab_size
        self.num_layer=num_layer
        self.num_header=num_header
        self.embeding_dim=embeding_dim
        self.input_length=input_length
        self.use_check_point=use_check_point
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
        self.tokenizer.padding_side ='right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.special_token=torch.tensor([73440]).long().reshape(1,1)

        self.embeding=Embeding(
            vocab_size=self.vocab_size,
            embeding_dim=self.embeding_dim
            )

        self.output_layer=Output_Layer(
            input_dim=self.embeding_dim,
            mid_dim=self.embeding_dim*4,
            output_dim=self.vocab_size
            )

        self.position_embeding=Position_Embeding(self.embeding_dim,self.input_length)

        self.attention_layer=nn.Sequential()

        for i in range(self.num_layer):
            self.attention_layer.add_module(f"attention{i}",Attention_Layer(
                use_group_attention=use_group_attention,
                group_num=group_num,
                input_dim=self.embeding_dim,
                num_heads=self.num_header,
                embeding_dim=self.embeding_dim//self.num_header,
                output_dim=self.embeding_dim,
                norm_type='L',
                use_check_point=self.use_check_point
            ))

            self.attention_layer.add_module(f"ffn{i}",Feedforward(
                input_dim=self.embeding_dim,
                mid_dim=self.embeding_dim,
                output_dim=self.embeding_dim,
                use_check_point=self.use_check_point,
                norm_type='L'
            ))
        
        self.decoder=nn.Sequential(
            self.embeding,
            self.attention_layer,
            self.output_layer
        )


    def get_ans(self,inputs):
        ans=self.decoder(inputs)
        return ans

        

    def loss(self,prediction,ans,loss_mask):
        batch_size=prediction.size()[0]
        length=prediction.size()[1]

        ans_index=[i for i in range(1,length)]
        ans=torch.index_select(ans,1,torch.tensor(ans_index).to(os.environ.get('device')))
        ans=torch.concat([ans,self.special_token.expand(batch_size,-1).to(os.environ.get('device'))],dim=1)

        mask=F.one_hot(ans,self.vocab_size)
        loss_mask=loss_mask.reshape(batch_size,length,1).expand(-1,-1,self.vocab_size)

        loss=-mask*loss_mask*torch.log(prediction)

        return loss.sum()/(batch_size*length)

    def use_tokenizer(self,inputs):
        outputs = self.tokenizer(inputs, return_tensors="pt",padding=True,max_length=self.input_length, truncation=True,add_special_tokens=True)
        return outputs

    
        


        

        
        



        
        
        

