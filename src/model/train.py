'''
Model training main function
Usage: 
    python -m model.train --cfg config.json
    python -m model.train --cfg '{...}'
Parameters:
    --cfg: hyperparameter configuration json file path or json string. 
'''

import argparse
import json
import os
import datetime
import time
# custom modules
from model.loader import Loader, load_loader
from model.generator import get_generator
from model.model import load_model
from core.util import GlobalSettings

gs = GlobalSettings()

def main() -> None:
    # load config file
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--cfg', type=str, 
                        default='config.json',
                        help='hyperparameter configuration json file path or json string')
    args = parser.parse_args()
    cfg = args.cfg
    if os.path.isfile(cfg):
        with open(cfg, 'r') as f:
            opt = json.load(f)
    else:
        try:
            opt = json.loads(cfg)
        except Exception as e:
            raise ValueError(f'cfg is not a valid file path or invalid json format: {e}')

    # create output
    name = opt.get('name', '')
    derived_folder = gs.get('derived_folder')
    out_folder = get_next_job_folder(derived_folder, name)
    
    # save config to output folder
    with open(os.path.join(out_folder, 'config.json'), 'w') as f:
        json.dump(opt, f, indent=4)
    
    # load data
    loader_opt = opt['loader']
    loader_opt['file_list'] = os.path.join(gs.get('data_folder'), loader_opt['file_list'])
    loader = load_loader(loader_opt)
    train_df = loader.train_df
    if loader_opt['kfold'] > 1:
        val_df = loader.val_df
    else:
        val_df = loader.test_df

    # create generator
    generator_opt = opt['generator']
    sampling = 'stratified'
    if 'n' in generator_opt:
        n = generator_opt['n']
        del generator_opt['n']
    else:
        n = None
    category = 'label'
    # compute minimum of number of samples per class
    value_counts = train_df[category].value_counts()
    min_sample_per_class = value_counts.min()
    n_class = len(value_counts) # number of classes
    if n is None:
        # use all data
        print(f'All {len(train_df)} samples are used for training')
    elif n is not None and n // n_class < min_sample_per_class:
        n_per_class = n // n_class
        idx = Loader.stratified_sample(train_df, category, n_per_class).index.tolist()
        train_df = train_df.loc[idx]
        print(f'Stratified sample {n_per_class} samples per class, total {n} samples are used for training')
    else:
        raise ValueError(f'Invalid n: {n} for {n_class} classes for stratified sampling, minimum sample per class is {min_sample_per_class}')
    train_generator, val_generator = get_generator(train_df, val_df, generator_opt)

    # create model
    model_opt = opt['model']
    model_opt['out_folder'] = out_folder
    model_class = load_model(model_opt)

    # train
    model_class.train(train_generator, val_generator)
    return

def get_next_job_folder(root_folder: str, name: str='') -> str:
    '''
    create a new folder under root_folder with datetime string as folder name
    name: optional notes in the folder name
    '''
    # create datetime string as job ID e.g. 20231118123456
    id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if name == '':
        out_folder = os.path.join(root_folder, id)
    else:
        out_folder = os.path.join(root_folder, id + '_' + name)
    while os.path.exists(out_folder):
        time.sleep(1) # prevent job ID conflict
        id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if name == '':
            out_folder = os.path.join(root_folder, id)
        else:
            out_folder = os.path.join(root_folder, id + '_' + name)
    os.makedirs(out_folder, exist_ok=False)
    return out_folder

if __name__ == '__main__':
    main()
