from cdfsl import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

def make_cdfsl_loader(dataset, epoch_len, n_shot = 1, k_way = 5, q_query = 15, small = False):
  # conver dataset string to class
  dataset = {
    'isic': ISIC_few_shot,
    'eurosat': EuroSAT_few_shot,
    'cropdisease': CropDisease_few_shot,
    'chest': Chest_few_shot,
  }[dataset.lower()]
  
  data_args = {
    'n_way': k_way,
    'n_support': n_shot,
    'n_query': q_query,
    'n_eposide': epoch_len, # FIXME typo!
  }
  
  datamgr = dataset.SetDataManager(84 if small else 224, **data_args)
  return datamgr.get_data_loader(aug = False)


if __name__ == '__main__':
  few_shot_params = dict(n_way = 5, n_support = 1) 
  image_size = 224
  iter_num = 600

  #for dataset in [ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot]:
  for dataset in [EuroSAT_few_shot]:
    print('probe', dataset)
    datamgr = dataset.SetDataManager(image_size, 
                                     n_eposide = iter_num, 
                                     n_query = 15, 
                                     **few_shot_params)
    loader = datamgr.get_data_loader(aug = False)
    
    for i, (x, y) in enumerate(loader):
      print(x.shape, y.shape)
      print(y)
      
      if i > 3:
        break
