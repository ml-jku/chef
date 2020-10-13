import torch
from torch import nn

import argparse
import numpy as np
import json
from types import SimpleNamespace

from cdfsl import make_cdfsl_loader
from core import EvaluateFewShot
from datasets import MiniImageNet
from hebb import stage_two, hebb_rule
from res12 import resnet12
from res10 import res10
from models import conv64

from utils import prepare_meta_batch, make_task_loader


class ModelWrapper(nn.Module):
  def __init__(self, embed, fc_sizes):
    super(ModelWrapper, self).__init__()
    self.embed = embed
    self.feature_index = [-1]
    
    seq = [] #[nn.ReLU()]
    
    for i in range(len(fc_sizes)-2):
      seq += [nn.Linear(fc_sizes[i], fc_sizes[i+1]), nn.ReLU(), nn.Dropout(0.5)]
    
    seq += [nn.Linear(fc_sizes[-2], fc_sizes[-1])]
    self.output_layer = nn.Sequential(*seq)
  
  def forward(self, x, output_layer=True):
    self.x = [self.embed(x)]
    
    for m in self.output_layer:
      self.x += [m(self.x[-1].flatten(1))]
    
    self.features = [self.x[fi] for fi in self.feature_index]
    
    return self.x[-1]


parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('config', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--n', type=int)
parser.add_argument('--k', type=int)
parser.add_argument('--q', type=int)
parser.add_argument('--eval-batches', type=int)
parser.add_argument('--gpu', type=int, nargs='+')
parser.add_argument('--num-workers', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--dropout', type=float)
parser.add_argument('--hebb-lr', type=float)
parser.add_argument('--inner-val-steps', type=int)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--backbone', choices=['conv64', 'res12', 'res10', 'res18'])
parser.add_argument('--seed', type=int)
parser.add_argument('--feature-index', type=int, nargs='+')
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

eval_few_shot_args = {
  'num_tasks': args.eval_batches,
  'n_shot': args.n,
  'k_way': args.k,
  'q_queries': args.q,
  'prepare_batch': prepare_meta_batch(
      args.n, args.k, args.q, args.meta_batch_size, 2, device),
  'inner_train_steps': args.inner_val_steps,
  'hebb_lr': args.hebb_lr,
  'device': device,
  'xdom': hasattr(args, 'dataset'),
}

state_dict = torch.load(args.model, map_location=device)


if args.backbone == 'res12':
  embed = resnet12(avg_pool=False, drop_rate=args.dropout, dropblock_size=5)
  fc_sizes = [16000, 4000, 1000, 80]
elif args.backbone == 'res10':
  embed = res10()
  fc_sizes = [25088, 4000, 1000, 80]
elif args.backbone == 'conv64':
  embed = conv64()
  fc_sizes = [1600, 400, 100, 80]

model = ModelWrapper(embed, fc_sizes)
model.feature_index = args.feature_index #[-1, -2, -4, -5, -7, -8]
model = nn.DataParallel(model, device_ids=args.gpu)
model.load_state_dict(state_dict)
model = model.to(device, dtype=torch.double)
model.eval()

if hasattr(args, 'dataset'):
  test_loader = make_cdfsl_loader(args.dataset, 
                                  args.eval_batches, 
                                  args.n, 
                                  args.k, 
                                  args.q, 
                                  small=(args.backbone!='res10'))
else:
  test_loader = make_task_loader(MiniImageNet('test', 
                                              small=(args.backbone!='res10')), 
                                 args, train=False, meta=True)

loss_fn = nn.CrossEntropyLoss().to(device)

evaluator = EvaluateFewShot(eval_fn=hebb_rule, 
                            taskloader=test_loader,
                            **eval_few_shot_args)

#logs = {'dummy': 0} # it's important to have logs be non-empty
logs = {
  'dataset': args.dataset if hasattr(args, 'dataset') else 'miniImagenet',
  'n-shot': args.n,
  'feature_index': args.feature_index,
  'hebb_lr': args.hebb_lr,
  'inner_val_steps': args.inner_val_steps,
}
evaluator.model = {'sys1': model}
evaluator.optimiser = None
evaluator.loss_fn = loss_fn
evaluator.on_epoch_end(0, logs)
print(logs)



