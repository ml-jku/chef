import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse
import numpy as np
import datetime
import os
import json
from types import SimpleNamespace

from datasets import MiniImagenetHorizontal
from res12 import resnet12
from res10 import res10
from models import conv64
from utils import adjust_learning_rate

class ModelWrapper(nn.Module):
  def __init__(self, embed, fc_sizes):
    super(ModelWrapper, self).__init__()
    self.embed = embed
    
    seq = []
    
    for i in range(len(fc_sizes)-2):
      seq += [nn.Linear(fc_sizes[i], fc_sizes[i+1]), nn.ReLU(), nn.Dropout(0.5)]
    
    seq += [nn.Linear(fc_sizes[-2], fc_sizes[-1])]
    self.output_layer = nn.Sequential(*seq)
  
  def forward(self, x):
    x = self.embed(x)
    return self.output_layer(torch.relu(x))


# CAUTION: must not use default values as they would override the json config
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('config', type=str)
#parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--weight-decay', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--gpu', type=int, nargs='+')
parser.add_argument('--seed', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--backbone', choices=['conv64', 'res12', 'res10'])
parser.add_argument('--num-workers', type=int)
#parser.add_argument('--relu-out', action='store_true')
#parser.add_argument('--flag', type=bool)
args = parser.parse_args()

with open(args.config) as f:
  config = json.load(f)

# override config with cmd line args
config.update(vars(args))
args = SimpleNamespace(**config)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
assert(torch.cuda.is_available())
device = torch.device(args.gpu[0])
train_data = MiniImagenetHorizontal('train', small=(args.backbone!='res10'))
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.num_workers, drop_last=True)
val_data = MiniImagenetHorizontal('val', small=(args.backbone!='res10'))
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, 
                        num_workers=args.num_workers, drop_last=False)
runid = datetime.datetime.now().strftime('%y%m%dT%H%M%S') + f'P{os.getpid()}'
print(f'runid={runid}')

if args.backbone == 'res12':
  embed = resnet12(avg_pool=False, drop_rate=args.dropout, dropblock_size=5)
  fc_sizes = [16000, 4000, 1000, len(train_data.classes)]
elif args.backbone == 'res10':
  embed = res10()
  fc_sizes = [25088, 4000, 1000, len(train_data.classes)]
elif args.backbone == 'conv64':
  embed = conv64()
  fc_sizes = [1600, 400, 100, len(train_data.classes)]


model = ModelWrapper(embed, fc_sizes)
model = model.to(device)
model = nn.DataParallel(model, device_ids=args.gpu)
lossf = nn.CrossEntropyLoss().to(device)
#optim = torch.optim.Adam(model.parameters(), lr=args.lr)
optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                        weight_decay=args.weight_decay)

best_loss = float('inf')

for epoch in range(args.epochs):
  adjust_learning_rate(optim, epoch, args.lr)
  model.train()
  
  for i, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    loss = lossf(y_hat, y)
    acc = (y_hat.argmax(dim=1) == y).float().mean()
    print(f'\33[2K\repoch: {epoch}/{args.epochs} iter: {i}/{len(train_loader)} ' + \
          f'loss: {loss.item():.4f} acc: {acc.item()*100:.2f}%', 
          end='')
    optim.zero_grad()
    loss.backward()
    optim.step()
  
  model.eval()
  loss = []
  acc = []
  
  with torch.no_grad():
    for x, y in val_loader:
      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      loss.append(lossf(y_hat, y))
      acc.append((y_hat.argmax(dim=1) == y).float().mean())
  
  loss = sum(loss) / len(loss)
  acc  = sum(acc) / len(acc)
  post = ''
  
  if loss < best_loss:
    torch.save(model.state_dict(), f'{runid}_{args.backbone}.pth')
    best_loss = loss
    post = 'model saved'
  
  print(f'\33[2K\repoch: {epoch+1}/{args.epochs} iter: 1/1 ' + \
        f'loss: {loss:.4f} acc: {acc*100:.2f}% ' + post)






