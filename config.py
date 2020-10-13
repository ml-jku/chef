import os


PATH = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir('/local00/bioinf/miniImageNet'):
  DATA_PATH = '/local00/bioinf/'
elif os.path.isdir('/iarai/work/metalearn/data/miniImageNet'):
  DATA_PATH = '/iarai/work/metalearn/data/'
elif os.path.isdir('/publicdata/miniImageNet'):
  DATA_PATH = '/publicdata/'

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')

_TIERED_IMAGENET_DATASET_DIR = None

for root in ['/local00/bioinf', '/iarai/work/metalearn/data', '/publicdata']:
  path = os.path.join(root, 'tiered_imagenet')
  
  if os.path.isdir(path):
    _TIERED_IMAGENET_DATASET_DIR = path
    break

# set CD-FSL paths

subdirs = ['ISIC', 'nih-chest-xrays', 'plant-disease', 'EuroSAT/2750']
varnames = ['ISIC_path', 'ChestX_path', 'CropDisease_path', 'EuroSAT_path']

for sub, var in zip(subdirs, varnames):
  for base in ('/local00/bioinf/', '/publicdata/'):
    path = os.path.join(base, sub)
    
    if os.path.isdir(path):
      exec(f'{var} = "{path}"')
      break

