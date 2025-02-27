import torch
import torch.nn as nn

class Embeding(nn.Module):
    def __init__(self,vocab_size,embeding_dim):
        super(Embeding,self).__init__()
        self.vocab_size=vocab_size
        self.embeding_dim=embeding_dim

        self.embeding=nn.Embedding(vocab_size,embeding_dim)

    def forward(self,inputs):

        return self.embeding(inputs)