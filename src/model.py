"""
model architecture definition and creator
"""

import torch
from torch import nn

class WeightDrop(torch.nn.Module):

    def __init__(self, module, weights, dropout=0):
      super(WeightDrop, self).__init__()
      self.module = module
      self.weights = weights
      self.dropout = dropout
      self._setup()
    
    def widget_demagnetizer_y2k_edition(*args, **kwargs):
      return
    
    def _setup(self):
      if issubclass(type(self.module), torch.nn.RNNBase):
        self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition
    
        for name_w in self.weights:
          print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
          w = getattr(self.module, name_w)
          del self.module._parameters[name_w]
          self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))
    
    def _setweights(self):
      for name_w in self.weights:
        raw_w = getattr(self.module, name_w + '_raw')

        mask = torch.nn.functional.dropout(torch.ones_like(raw_w), 
                                           p=self.dropout,
                                           training=True) * (1 - self.dropout)
        setattr(self.module, name_w, raw_w*mask)
    
    
    def forward(self, *args):
      self._setweights()
      return self.module.forward(*args)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
      mask = embed.weight.data.new().resize_(
          (embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
          embed.weight) / (1 - dropout)
      masked_embed_weight = mask * embed.weight
    else:
      masked_embed_weight = embed.weight
    if scale:
      masked_embed_weight = scale.expand_as(masked_embed_weight)*masked_embed_weight
    
    padding_idx = embed.padding_idx
    if padding_idx is None:
      padding_idx = -1
    
    embedding = torch.nn.functional.embedding(words, masked_embed_weight,
                                              padding_idx, embed.max_norm,
                                              embed.norm_type, 
                                              embed.scale_grad_by_freq,
                                              embed.sparse)
    return embedding


class LockedDropout(nn.Module):
    def __init__(self):
      super(LockedDropout, self).__init__()
    
    def forward(self, x, dropout):
      if not self.training or not dropout:
        return x
      m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
      mask = m.requires_grad_(False) / (1 - dropout)
      mask = mask.expand_as(x)
      return mask * x
  
    
class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                  dropoute=0.2, dropouti=0.2, dropouth=0.2, dropouto=0.2, 
                  weight_drop=0.5):
      super().__init__()
      self.num_layers = num_layers
      self.hidden_dim = hidden_dim
      self.embedding_dim = embedding_dim
    
      self.embedding = nn.Embedding(vocab_size, embedding_dim)
      self.embedding.weight.data.uniform_(-0.1, 0.1)
      # self.dropout = nn.Dropout(p=dropout_embd)
    
      self.lstms = []
      self.lstms.append(nn.LSTM(embedding_dim, hidden_dim, num_layers=1, 
                                dropout=0, batch_first=False))
      self.lstms.append(nn.LSTM(hidden_dim, hidden_dim, num_layers=1, 
                                dropout=0, batch_first=False))
      self.lstms.append(nn.LSTM(hidden_dim, embedding_dim, num_layers=1, 
                                dropout=0, batch_first=False))
      if not weight_drop:
        self.lstms = [WeightDrop(lstm, ['weight_hh_l0'],
                                 dropout=weight_drop) for lstm in self.lstms]
      self.lstms = nn.ModuleList(self.lstms)
    
      self.fc = nn.Linear(embedding_dim, vocab_size)
    
      self.fc.weight = self.embedding.weight
    
      self.lockdrop = LockedDropout()
      self.dropoute = dropoute
      self.dropouti = dropouti
      self.dropouth = dropouth
      self.dropouto = dropouto
    
    def forward(self, src):
      # embedding = self.dropout(self.embedding(src))
      embedding = embedded_dropout(self.embedding, src, 
                                   dropout=self.dropoute if self.training else 0)
      embedding = self.lockdrop(embedding, self.dropouti)
    
      # new_hiddens = []
      for l, lstm in enumerate(self.lstms):
        embedding, _ = lstm(embedding)
        if l != self.num_layers-1:
          embedding = self.lockdrop(embedding, self.dropouth)
    
      embedding = self.lockdrop(embedding, self.dropouto)
    
      prediction = self.fc(embedding)
      return prediction
  
if __name__ == '__main__':
    model = LanguageModel(vocab_size=2000, embedding_dim=300,
                          hidden_dim=1100, num_layers=2,
                          dropoute=0.2, dropouti=0.2,
                          dropouth=0.2, dropouto=0.2,weight_drop=0)
    print(model)
