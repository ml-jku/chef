# cd /publicdata/tiered_imagenet
# mkdir trainval && cd trainval
# for d in $(ls ../train); do ln -s ../train/$d $d; done
# for d in $(ls ../val); do ln -s ../val/$d $d; done

import os
import numpy as np

np.random.seed(0)
os.chdir('/local00/bioinf/tiered_imagenet')
#os.chdir('/publicdata/tiered_imagenet')

os.system('rm -rf hsplit_*')
os.mkdir('hsplit_train')
os.mkdir('hsplit_val')

# make class dirs
for root, dirs, files in os.walk('trainval', followlinks=True):
  for d in dirs:
    os.mkdir(os.path.join('hsplit_train', d))
    os.mkdir(os.path.join('hsplit_val', d))

# place file links in class dirs
for folder, dirs, files in os.walk('trainval/', topdown=True, followlinks=True):
  if len(dirs) > 0:
    dirs.sort()
    continue
  
  files.sort()
  permutation = np.random.permutation(len(files))
  
  for i, p in enumerate(permutation):
    # split 85% training 15% validation
    root = 'hsplit_' + ('train' if i < len(files) * 0.85 else 'val')
    link_name = os.path.join(root, folder.split('/')[1], files[p])
    target = os.path.join('../..', folder, files[p])
    os.system(f'ln -s {target} {link_name}')



