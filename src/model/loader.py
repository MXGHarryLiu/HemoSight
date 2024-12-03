import os
from typing import Tuple, Union, List
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

class Loader():
    def __init__(self):
        self._df = None # entire dataset cache: path, img_name, label, rel_path
        self._train_df = None
        self._val_df = None
        self._test_df = None

    @property
    def train_df(self) -> pd.DataFrame:
        if self._train_df is None:
            raise ValueError('Run split() first.')
        return self._train_df.copy()
    
    @property
    def val_df(self) -> pd.DataFrame:
        if self._val_df is None:
            raise ValueError('Run split() first.')
        return self._val_df.copy()
    
    @property
    def test_df(self) -> pd.DataFrame:
        if self._test_df is None:
            raise ValueError('Run split() first.')
        return self._test_df.copy()

    def _stratified_kfold_split(self, df: pd.DataFrame, kfold: int, 
                                i_kfold: int, split_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        df: dataframe with label column
        kfold: number of folds, greater than 1
        i_kfold: index of fold to use as validation
        '''
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=split_seed)
        q = skf.split(df, df['label'])
        train_index, val_index = list(q)[i_kfold]
        val_df = df.iloc[val_index]
        train_df = df.iloc[train_index]
        return train_df, val_df

    def load(self, file_list: str) -> None:
        '''
        load image data from reference csv
        file_list: path to csv file
        '''
        csv_df = pd.read_csv(file_list)
        # convert to categorical type
        df = csv_df[['rel_path', 'img_name']].copy()
        df['label'] = csv_df['label'].astype('category')
        # expand relative path to absolute path
        folder = os.path.dirname(file_list)
        df['path'] = [os.path.join(folder, file) for file in df['rel_path']]
        self._df = df
        return
    
    def split(self, split_seed: int=0, 
              test_ratio: float=0.1, 
              kfold: int=5, 
              i_kfold: int=0) -> None:
        '''
        split data into train, validation and test set. Run load() first. 
        split_seed: rng seed
        test_ratio: test set radio
        kfold: validation fold count, if 1, use all train data, and no validation
        i_kfold: validation fold index to use, 0-base
        '''
        # split
        df = self._df
        if df is None:
            raise ValueError('Run load() first.')
        train_full_df, test_df = train_test_split(df, random_state=split_seed, 
                                             test_size=test_ratio, 
                                             stratify=df['label'])
        print('Split {} images to {} train and {} test images'.format(
            len(df), len(train_full_df), len(test_df)))
        if kfold == 1:
            train_df = train_full_df
            val_df = None
        else:
            train_df, val_df = self._stratified_kfold_split(train_full_df, kfold, i_kfold, split_seed)
            print('Split {} images to {} train and {} validation images'.format(
                len(train_full_df), len(train_df), len(val_df))) 
        # cache
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df
        return
    
    @staticmethod
    def stratified_sample(df: pd.DataFrame, column: str, n_sample: int) -> pd.DataFrame:
        '''
        df: dataframe
        column: column name to stratify
        n_sample: number of samples per class
        '''
        def sample(x, n):
            if len(x) <= n:
                Warning('Class {} has only {} samples.'.format(x.iloc[0][column], len(x)))
                return x
            return x.sample(n, random_state=0, replace=False)
        df_sample = df.groupby(column, group_keys=False).apply(lambda x: sample(x, n_sample))
        return df_sample
    
    @staticmethod
    def load_morph(train_df, test_df, csv_path: str, 
               columns: Union[List[str], str, None]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        append morphology columns to train_df and test_df
        columns: list of columns to keep, None means keep all columns
        '''
        morph_df = pd.read_csv(csv_path)
        # drop img_name, label
        morph_df.drop(columns=['img_name', 'label'], inplace=True)
        # drop columns not in columns, except path
        if columns is not None:
            if type(columns) == str:
                columns = [columns]
            if len(columns) == 0:
                print('Warning: columns is empty, no category will be kept')
            columns = set(columns + ['path'])
            morph_df = morph_df[[col for col in morph_df.columns if col in columns]]
        # rename path to rel_path, rel_path is the key
        morph_df.rename(columns={'path': 'rel_path'}, inplace=True)
        # convert all except rel_path to categorical
        morph_df = morph_df.apply(lambda x: x.astype('category') if x.name != 'rel_path' else x)
        # merge with train_df, missing values are filled with NaN
        train_index = train_df.index
        train_df = pd.merge(train_df, morph_df, on='rel_path', how='left')
        train_df.set_index(train_index, inplace=True)
        # merge with test_df
        test_index = test_df.index
        test_df = pd.merge(test_df, morph_df, on='rel_path', how='left')
        test_df.set_index(test_index, inplace=True)
        return train_df, test_df

def load_loader(opt: dict) -> Loader:
    loader = Loader()
    loader.load(file_list=opt['file_list'])
    loader.split(split_seed=opt['split_seed'], 
              test_ratio=opt['test_ratio'], 
              kfold=opt['kfold'], 
              i_kfold=opt['i_kfold'])
    return loader
