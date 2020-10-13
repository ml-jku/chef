from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle


from config import DATA_PATH, _TIERED_IMAGENET_DATASET_DIR


class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset, small=True):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation', 'train', 'val', 'test', 'trainval'):
            raise(ValueError, 'subset must be one of (background, evaluation, train, val, test)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if subset in ('train', 'trainval'):
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84 if small else 224), #(224), #(84),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif subset in ('val', 'test'):
            self.transform = transforms.Compose([
                transforms.Resize(96 if small else 256),
                transforms.CenterCrop(84 if small else 224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else: # subset in ('background', 'evaluation')
            self.transform = transforms.Compose([
                # FIXME this is Knagg's preprocessing
                transforms.CenterCrop(224),
                transforms.Resize(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __getitem__(self, item):
        # TODO raise IndexError if item out of range
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset), followlinks=True):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset), followlinks=True):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImagenetHorizontal(Dataset):
    def __init__(self, split, small=True):
        self.path = os.path.join(DATA_PATH, f'miniImageNet/hsplit_{split}')
        self.files = []
        
        for root, dirs, files in os.walk(self.path):
            if len(dirs) > 0:
                self.classes = sorted(dirs)
            elif len(files) > 0:
                c = root.split('/')[-1]
                self.files += [os.path.join(c, f) for f in files]
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84 if small else 224),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif split == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(96 if small else 256),
                transforms.CenterCrop(84 if small else 224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    
    def __getitem__(self, item):
        f = self.files[item]
        class_name = f.split('/')[0]
        label = self.classes.index(class_name)
        img = os.path.join(self.path, f)
        img = Image.open(img)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.files)


class TieredImagenet(Dataset):
    """
    follow instructions at https://github.com/kjunelee/MetaOptNet
    download https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG
    dataset class adapted from https://github.com/kjunelee/MetaOptNet/blob/master/data/tiered_imagenet.py
    """
    @staticmethod
    def buildLabelIndex(labels):
        label2inds = {}
        for idx, label in enumerate(labels):
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

        return label2inds
    
    @staticmethod
    def load_data(file):
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data
    
    
    def __init__(self, phase='train', do_not_use_random_transf=False):
        assert(phase=='train' or phase=='val' or phase=='test')
        self.phase = phase
        self.name = 'tieredImageNet_' + phase

        print('Loading tiered ImageNet dataset - phase {0}... '.format(phase), end='', flush=True)
        file_train_categories_train_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images.npz')
        label_train_categories_train_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')
        file_train_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images.npz')
        label_train_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')
        file_train_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_images.npz')
        label_train_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'train_labels.pkl')

        file_val_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'val_images.npz')
        label_val_categories_val_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'val_labels.pkl')
        file_test_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'test_images.npz')
        label_test_categories_test_phase = os.path.join(
            _TIERED_IMAGENET_DATASET_DIR,
            'test_labels.pkl')
        
        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = self.load_data(label_train_categories_train_phase)
            #self.data = data_train['data']
            self.labels = data_train['labels']
            self.data = np.load(file_train_categories_train_phase)['images']#np.array(self.load_data(file_train_categories_train_phase))
            #self.labels = self.load_data(file_train_categories_train_phase)#data_train['labels']

            self.label2ind = self.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = self.load_data(label_train_categories_test_phase)
                data_base_images = np.load(file_train_categories_test_phase)['images']
                
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = self.load_data(label_test_categories_test_phase)
                data_novel_images = np.load(file_test_categories_test_phase)['images']
            else: # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = self.load_data(label_train_categories_val_phase)
                data_base_images = np.load(file_train_categories_val_phase)['images']
                #print (data_base_images)
                #print (data_base_images.shape)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = self.load_data(label_val_categories_val_phase)
                data_novel_images = np.load(file_val_categories_val_phase)['images']

            if False: # adaption by toto
                self.data = np.concatenate(
                    [data_base_images, data_novel_images], axis=0)
                self.labels = data_base['labels'] + data_novel['labels']
            else:
                self.data = data_novel_images
                self.labels = data_novel['labels']

            self.label2ind = self.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            if False: # adaption by toto
                self.labelIds_base = self.buildLabelIndex(data_base['labels']).keys()
                self.labelIds_novel = self.buildLabelIndex(data_novel['labels']).keys()
                self.num_cats_base = len(self.labelIds_base)
                self.num_cats_novel = len(self.labelIds_novel)
                intersection = set(self.labelIds_base) & set(self.labelIds_novel)
                #print (intersection)
                assert(len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        
        print('done')

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
    
    def num_classes(self):
        return self.num_cats


class TieredImagenetHorizontal(Dataset):
    def __init__(self, split, small=True):
        self.path = os.path.join(DATA_PATH, f'tiered_imagenet/hsplit_{split}')
        self.files = []
        
        for root, dirs, files in os.walk(self.path):
            if len(dirs) > 0:
                self.classes = sorted(dirs)
            elif len(files) > 0:
                c = root.split('/')[-1]
                self.files += [os.path.join(c, f) for f in files]
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84 if small else 224),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        elif split == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(96 if small else 256),
                transforms.CenterCrop(84 if small else 224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    
    def __getitem__(self, item):
        f = self.files[item]
        class_name = f.split('/')[0]
        label = self.classes.index(class_name)
        img = os.path.join(self.path, f)
        img = Image.open(img)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.files)


class ImagenetBasedDataset(Dataset):
    def __init__(self, split, small=True, tier=False, horizontal=False):
        base = 'tiered_imagenet' if tier else 'miniImageNet'
        sub = ('hsplit_' if horizontal else ('' if tier else 'images_'))
        
        self.path = os.path.join(DATA_PATH, base, sub + split)
        self.files = []
        self.label2ind = {}
        
        for root, dirs, files in os.walk(self.path):
            if len(dirs) > 0:
                self.classes = sorted(dirs)
            elif len(files) > 0:
                c = root.split('/')[-1]
                base = len(self.files)
                self.label2ind[c] = range(base, base + len(files))
                self.files += [os.path.join(c, f) for f in files]
        
        self.labelIds = self.classes
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84 if small else 224),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else: # i.e. split == 'val' or split == 'test'
            self.transform = transforms.Compose([
                transforms.Resize(96 if small else 256),
                transforms.CenterCrop(84 if small else 224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    
    def __getitem__(self, item):
        f = self.files[item]
        class_name = f.split('/')[0]
        label = self.classes.index(class_name)
        img = os.path.join(self.path, f)
        img = Image.open(img)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.files)



class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)


if __name__ == '__main__':
    for split in ('train', 'val', 'test'):
      for small in (True, False):
        for tier in (True, False):
          for horizontal in (True, False):
            if split == 'test' and horizontal:
              continue
            
            print('split', split, 'small', small, 'tier', tier, 'horizontal', horizontal)
            ds = ImagenetBasedDataset(split, small, tier, horizontal)
            print(len(ds))
            img, label = ds[0]
            img, label = ds[-1]
            img, label = ds[len(ds) // 2]
    

