import torch
from torch import nn
import torchvision.models
from torchvision.models.resnet import BasicBlock, Bottleneck
import argparse
import numpy as np
import json
import sys
from types import SimpleNamespace

from cdfsl import make_cdfsl_loader
from core import EvaluateFewShot
from datasets import ImagenetBasedDataset, MiniImageNet
from hebb import stage_two, hebb_rule
from res12 import resnet12
from res10 import res10
from models import conv64

from utils import prepare_meta_batch, make_task_loader


def basic_block_forward(layer, x):
  identity = x[-1]
  x.append(layer.conv1(x[-1]))
  x.append(layer.bn1(x[-1]))
  x.append(layer.relu(x[-1]))
  x.append(layer.conv2(x[-1]))
  x.append(layer.bn2(x[-1]))

  if layer.downsample is not None:
    identity = layer.downsample(identity)

  x.append(x[-1] + identity)
  #print('basic_block_forward', len(x)-1)
  x.append(layer.relu(x[-1]))
  return x


def bottleneck_forward(layer, x):
  identity = x[-1]
  x.append(layer.conv1(x[-1]))
  x.append(layer.bn1(x[-1]))
  x.append(layer.relu(x[-1]))
  x.append(layer.conv2(x[-1]))
  x.append(layer.bn2(x[-1]))
  x.append(layer.relu(x[-1]))
  x.append(layer.conv3(x[-1]))
  x.append(layer.bn3(x[-1]))
  
  if layer.downsample is not None:
    identity = layer.downsample(identity)
  
  x.append(x[-1] + identity)
  x.append(layer.relu(x[-1]))
  return x


def recursive_forward(module, x):
  if isinstance(module, BasicBlock):
    x = basic_block_forward(module, x)
    return x
  elif isinstance(module, Bottleneck):
    x = bottleneck_forward(module, x)
    return x
  elif isinstance(module, nn.AdaptiveAvgPool2d):
    x.append(module.forward(x[-1]).flatten(1))
    return x
  elif hasattr(module, '_modules') and module._modules:
    for m in module._modules.values():
      x = recursive_forward(m, x)
    return x
  else:
    x.append(module.forward(x[-1]))
    return x


class ModelWrapper(nn.Module):
  def __init__(self, embed):
    super(ModelWrapper, self).__init__()
    self.embed = embed
    self.feature_index = [-1]
  
  def forward(self, x, output_layer=True):
    self.x = [x]
    self.x = recursive_forward(self.embed, self.x)
    self.features = [self.x[fi].flatten(1) for fi in self.feature_index]
    
    #[print(f.shape) for f in self.features]
    #exit()
    
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
parser.add_argument('--hebb-lr', type=float)
parser.add_argument('--inner-val-steps', type=int)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--feature-index', type=int, nargs='+')
# res18 ablation: -1 59 52 45 38 31
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
  'xdom': hasattr(args, 'dataset') and args.dataset not in ('mini', 'tier'),
}

model = torchvision.models.resnet18(pretrained=True)
#model = torchvision.models.resnet34(pretrained=True)
#model = torchvision.models.resnet50(pretrained=True)
#model = torchvision.models.resnet101(pretrained=True)
#model = torchvision.models.resnet152(pretrained=True)

#model_orig = model # FIXME integrity check
model = ModelWrapper(model)
model.feature_index = args.feature_index #[-1, -2, -3, -8]
model = nn.DataParallel(model, device_ids=args.gpu)
model = model.to(device, dtype=torch.double)
model.eval()

# FIXME integrity check
#model_orig = model_orig.to(device, dtype=torch.double)
#x = torch.rand(1, 3, 224, 244).to(device, dtype=torch.double)
#print((model(x)-model_orig(x)).sum())
#exit()


if (not hasattr(args, 'dataset')) or args.dataset == 'mini':
  #test_loader = make_task_loader(MiniImageNet('test', small=False), 
  #                               args, train=False, meta=True)
  test_loader = make_task_loader(ImagenetBasedDataset('test', small=False), 
                                 args, train=False, meta=True)
elif args.dataset == 'tier':
  test_loader = make_task_loader(ImagenetBasedDataset('test', small=False, tier=True), 
                                 args, train=False, meta=True)
else:
  test_loader = make_cdfsl_loader(args.dataset, 
                                  args.eval_batches, 
                                  args.n, 
                                  args.k, 
                                  args.q, 
                                  small=False)

loss_fn = nn.CrossEntropyLoss().to(device)

evaluator = EvaluateFewShot(eval_fn=hebb_rule, 
                            taskloader=test_loader,
                            **eval_few_shot_args)

#logs = {'dummy': 0} # it's important to have logs be non-empty
logs = {
  'dataset': args.dataset if hasattr(args, 'dataset') else 'miniImagenet',
  'feature_index': args.feature_index,
}
evaluator.model = {'sys1': model}
evaluator.optimiser = None
evaluator.loss_fn = loss_fn
evaluator.on_epoch_end(0, logs)

print(logs)
feature_index = 'ensemble' if len(args.feature_index) > 1 else args.feature_index
print(f'res18,{args.dataset},{args.n},{feature_index},{logs[evaluator.metric_name]}')



