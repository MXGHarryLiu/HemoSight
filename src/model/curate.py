import os
from typing import Tuple, List
import pandas as pd

class Curator():

    def __init__(self, tier1_folder: str) -> None:
        '''
        tier1_folder: full path to root data folder
        '''
        self._tier1_folder = tier1_folder
        return
    
    def _get_file(self, folder: str) -> List[str]:
        # obtain all jpg files recursively in tier2_folder
        root_folder = self._tier1_folder
        folder = os.path.join(root_folder, folder)
        file_list = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.jpg'):
                    file_list.append(os.path.join(root, file))
        # convert to relative path
        file_list = [os.path.relpath(file, start=root_folder) for file in file_list]
        # use / instead of \
        file_list = [file.replace('\\', '/') for file in file_list]
        return file_list

    def load_data(self, name: str, mapping: dict={}) -> pd.DataFrame:
        '''
        load data from folder
        input:
        name: dataset name
        mapping: dictionary of class name mapping, e.g. {''metamyelocyte': 'ig'}
        output fields: 
        rel_path: relative path to image
        img_name: image name without extension
        label: class name, use singular
        code: class code (sorted alphabetically)
        '''
        if name in ['20230910', '20230926', '20240520']:
            files = self._get_file(name)
            labels = []
            for file in files:
                # determine class from last folder name
                # 20230910\case 1\2315200480A blasts\*.jpg --> blasts
                class_name = file.split('/')[-2]
                # case_name = file.split('/')[-3]
                # case_id = int(case_name.split(' ')[-1])
                # if contains space or dash, take the last word
                if ' ' in class_name:
                    class_name = class_name.split(' ')[-1]
                else:
                    class_name = class_name.split('-')[-1]
                # standardize class name from plural to singular
                if class_name.endswith('s'):
                    class_name = class_name[:-1]
                labels.append(class_name)
        elif name == 'PBC_dataset_normal_DIB':
            files = self._get_file(name)
            labels = [os.path.basename(os.path.dirname(file)) for file in files]
        else:
            raise ValueError(f'Invalid name {name}. ')
        if len(mapping) > 0:
            labels = [mapping.get(label, label) for label in labels]
        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        df = pd.DataFrame({'rel_path': files, 'img_name': names, 'label': labels})
        df['label'] = df['label'].astype('category')
        df['code'] = df['label'].cat.codes
        classes = df['label'].cat.categories.values
        print(f'Loaded {len(df)} files in {len(classes)} classes. ')
        print(df['label'].value_counts().sort_index())
        return df
    
    def merge_data(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        '''
        merge dataframes
        '''
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df['label'] = df['label'].astype('category')
        df['code'] = df['label'].cat.codes
        classes = df['label'].cat.categories.values
        print(f'Merge {len(dfs)} datasets: total {len(df)} files in {len(classes)} classes. ')
        print(df['label'].value_counts().sort_index())
        return df
    
    def train_test_split(self, df: pd.DataFrame, test_size: float=0.1, 
                         random_state: int=1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        stratified train test split
        '''
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, 
                                             stratify=df['label'],
                                             random_state=random_state,
                                             shuffle=True)
        print(f'Split (stratified) {len(df)} files to {len(train_df)} train and {len(test_df)} test. ')
        return train_df, test_df
