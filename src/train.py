"""
This script handles data loading, model definition, training, validation, and checkpoint saving.
It accepts various command-line arguments to configure the training process.

Usage:
    python train.py --dataset /path/to/dataset --epochs 50 --batch_size 32
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

def parse_args():
    """Training Options for Language Modeling"""
    parser = argparse.ArgumentParser(description='Language Modeling - PyTorch')
    
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
    # dataset
    parser.add_argument('--dataset', type=str, default='data/wikitext-2',
                        help='dataset path (default: data/wikitext-2)')
    parser.add_argument('--seq_len', type=int, default=70,
                        help='sequence length (default: 70)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='input batch size for training (default: 80)')    
    parser.add_argument('--lr', type=float, default=7.5, metavar='LR',
                        help='learning rate (default: 7.5)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight_decay (default: 1e-6)')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='clip (default: 1e-6)')   
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save_folder', default='models',
                        help='Directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    # repeatability
    parser.add_argument('--seed', type=int, default=8,
                        help='seed, use None for turning off (default: 8)')

    # the parser
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # print(args)
    return args


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
    'train model for one epoch'
    model.train()
    loss_train = AverageMeter()
    metric.reset()
    
    with tqdm.tqdm(train_loader, unit='batch') as tepoch:
      for inputs, targets in tepoch:
        if epoch:
          tepoch.set_description(f'Epoch {epoch}')
    
        inputs = inputs.t().to(args.device)
        targets = targets.t().to(args.device)
    
        outputs = model(inputs)
    
        loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
    
        loss.backward()
    
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=args.clip)
    
        optimizer.step()
        optimizer.zero_grad()
    
        loss_train.update(loss.item(), n=len(targets))
        metric.update(outputs, targets)
    
        tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())
    
    return model, loss_train.avg, metric.compute().item()


def evaluate(model, test_loader, loss_fn, metric):
    model.eval()
    loss_eval = AverageMeter()
    metric.reset()
    
    with torch.inference_mode():
      for inputs, targets in test_loader:
        inputs = inputs.t().to(args.device)
        targets = targets.t().to(args.device)
    
        outputs = model(inputs)
    
        loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
        loss_eval.update(loss.item(), n=len(targets))
    
        metric(outputs, targets)
    
    return loss_eval.avg, metric.compute().item()

def train_model(model):
    global loss_fn, metric
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay, 
                          momentum=args.momentum)
    
    loss_train_hist = []
    loss_valid_hist = []
    
    metric_train_hist = []
    metric_valid_hist = []
    
    best_loss_valid = torch.inf
    epoch_counter = 1
    
    for epoch in range(1, args.epochs+1):
        model, loss_train, metric_train = train_one_epoch(model,
                                                          train_loader,
                                                          loss_fn,
                                                          optimizer,
                                                          metric,
                                                          epoch)
        # Validation
        loss_valid, metric_valid = evaluate(model,
                                            valid_loader,
                                            loss_fn,
                                            metric)
          
        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)
          
        metric_train_hist.append(metric_train)
        metric_valid_hist.append(metric_valid)
          
        if loss_valid < best_loss_valid:
          torch.save(model, args.save_folder + '/model.pt')
          best_loss_valid = loss_valid
          best_epoch = epoch_counter
          print('Model Saved!')
          
        print(f'Valid: Loss = {loss_valid:.4}, Metric = {metric_valid:.4}')
        print()
          
        epoch_counter += 1
    return model 
        
if __name__ == '__main__':
    args = parse_args()

    train_set, valid_set, test_set, vocab = read_dataset(args.dataset,args.seq_len)
    
    set_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False)

    # for inputs, targets in train_loader:
    #   print(inputs[0, 0], targets[0, 0])
    #   break

    model = LanguageModel(vocab_size=len(vocab), 
                          embedding_dim=args.embed_dim,
                          hidden_dim=args.hidden_dim,
                          num_layers=args.num_layers,
                          dropoute=args.dropoute, dropouti=args.dropouti,
                          dropouth=args.dropouth, dropouto=args.dropouto).to(args.device)
    if args.resume:
        model = torch.load(args.save_folder + '/model.pt')
    # print(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = tm.text.Perplexity().to(args.device)
    
    if not args.eval:
        model = train_model(model)

    model = torch.load(args.save_folder + '/model.pt')
    model.eval()

    loss_train, metric_train = evaluate(model, train_loader, loss_fn, metric)
    print(f"Metric train: {metric_train:.2f}")
    loss_valid, metric_valid = evaluate(model, valid_loader, loss_fn, metric)
    print(f"Metric Valid: {metric_valid:.2f}")
    loss_test, metric_test = evaluate(model, test_loader, loss_fn, metric)
    print(f"Metric Test: {metric_test:.2f}")