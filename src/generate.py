"""
Generates text using a pre-trained machine learning model. 
Requires input text and accepts max length, temperature as command-line args. 

Usage: 
    python generate.py --input_text "This is my input" --max_length 100 --temperature 0.7
"""

import os
import argparse
import tqdm

# from src.data import read_dataset
# from src.utilities import AverageMeter, num_trainable_params, set_seed, logger
# from src.model import LanguageModel
from data import read_dataset
from utilities import AverageMeter, num_trainable_params, set_seed, logger
from model import LanguageModel

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchmetrics as tm
from torchtext.data.utils import get_tokenizer


def parse_args():
    """Generating Options for Language Modeling"""
    parser = argparse.ArgumentParser(description='Language Modeling - PyTorch')
    
    #generate
    parser.add_argument('input', default='models',
                        help='input prompt')
    parser.add_argument('--max_seq_len', type=int, default=30,
                        help='max length of generated text (default: 30)')
    parser.add_argument('--temperture', type=float, default=0.5,
                        help='temperture of sampling next token (default: 0.5)')
    # model
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of lstm layers (default: 3)')
    parser.add_argument('--hidden_dim', type=int, default=1150,
                        help='lstm layers hidden dimension (default: 1150)') 
    parser.add_argument('--embed_dim', type=int, default=400,
                        help='embedding layer  dimension (default: 400)') 
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='embedding layer dropout (default: 0.1)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='lstms layer input dropout (default: 0.65)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='lstms hidden layer dropout (default: 0.3)')
    parser.add_argument('--dropouto', type=float, default=0.4,
                        help='lstms layer output dropout (default: 0.4)')
    parser.add_argument('--weight_drop', type=float, default=0,
                        help='weigth drop (default: 0)')
    parser.add_argument('--save_folder', default='models',
                        help='Directory for saving checkpoint models')
        
    # the parser
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # print(args)
    return args

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, seed=None):

  indices = vocab(tokenizer(prompt))
  itos = vocab.get_itos()

  for i in range(max_seq_len):
    src = torch.LongTensor(indices).to(args.device)
    with torch.no_grad():
      prediction = model(src)

    # Low values like 0.1 for temperature, Makes softmax like argmax more
    probs = torch.softmax(prediction[-1]/temperature, dim = 0)
    idx = vocab["<ukn>"]
    while idx == vocab["<ukn>"]:
      idx = torch.multinomial(probs, num_samples =1).item()
    indices.append(idx)
    prompt += " " + itos[idx]
    # print(prompt)

    if idx == vocab["."]:
      return prompt
  
if __name__ == '__main__':
    args = parse_args()
    
    tokenizer = get_tokenizer('basic_english')
    vocab = torch.load('vocab.pt')
    
    model = LanguageModel(vocab_size=len(vocab), 
                          embedding_dim=args.embed_dim,
                          hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers,
                          dropoute=args.dropoute, dropouti=args.dropouti,
                          dropouth=args.dropouth, dropouto=args.dropouto).to(args.device)
    model = torch.load(args.save_folder + '/model.pt')
    model.eval()
    
    text = generate(args.input, args.max_seq_len, args.temperture, model, tokenizer, vocab, seed=None)
    print(text)