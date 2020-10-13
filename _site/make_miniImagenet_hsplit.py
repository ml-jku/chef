import os
import numpy as np

np.random.seed(0)
os.chdir('/local00/bioinf/miniImageNet')
#os.chdir('/publicdata/miniImageNet')

os.system('rm -rf hsplit_*')
os.mkdir('hsplit_train')
os.mkdir('hsplit_val')

# make class dirs
for root, dirs, files in os.walk('images_trainval', followlinks=True):
  for d in dirs:
    os.mkdir(os.path.join('hsplit_train', d))
    os.mkdir(os.path.join('hsplit_val', d))

# place file links in class dirs
for folder, dirs, files in os.walk('images_trainval/', topdown=True, followlinks=True):
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



