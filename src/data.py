"""
this function read dataset and make a proper input for DataLoader
dataset path should have 3 file: 
        wiki.test.tokens
        wiki.train.tokens
        wiki.valid.tokens 
"""

import torch
from torch.utils.data import Dataset
from torchdata.datapipes.iter import IterableWrapper
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def data_process(raw_text_iter, seq_len):
    global vocab, tokenizer

    data = torch.cat([torch.LongTensor(vocab(tokenizer(line))) for line in raw_text_iter])
      
    M = len(data) // seq_len
    r = len(data) % seq_len
    
    data = torch.cat((data, torch.LongTensor([0]))) if r==0 else data
      
    inputs = data[:M*seq_len]
    targets = data[1:M*seq_len+1]
      
    inputs = inputs.reshape(-1, seq_len)
    targets = targets.reshape(-1, seq_len)      
    return inputs, targets


class LanguageModelDataset(Dataset):

    def __init__(self, inputs, targets):
      self.inputs = inputs
      self.targets = targets
    
    def __len__(self):
      return self.inputs.shape[0]
    
    def __getitem__(self, idx):
      return self.inputs[idx], self.targets[idx]


def read_dataset(dataset_path,seq_len):
    global vocab, tokenizer
    
    with open(dataset_path + '/wiki.train.tokens', 'r') as file:
        train_iter = file.read().splitlines()
    
    with open(dataset_path + '/wiki.test.tokens', 'r') as file:
        test_iter = file.read().splitlines()
    
    with open(dataset_path + '/wiki.valid.tokens', 'r') as file:
        valid_iter = file.read().splitlines()
    
    train_iter = IterableWrapper(train_iter)
    test_iter = IterableWrapper(test_iter)
    valid_iter = IterableWrapper(valid_iter)
    
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter),
                                      specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    torch.save(vocab, 'vocab.pt')
     
    X_train, y_train = data_process(train_iter, seq_len)
    X_valid, y_valid = data_process(valid_iter, seq_len)
    X_test, y_test = data_process(test_iter, seq_len)

    train_set = LanguageModelDataset(X_train, y_train)
    valid_set = LanguageModelDataset(X_valid, y_valid)
    test_set = LanguageModelDataset(X_test, y_test)
    return train_set, valid_set, test_set, vocab
    
