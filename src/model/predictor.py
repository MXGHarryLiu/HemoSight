import os
import joblib
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pydantic import validate_call
import openTSNE
from sklearn.pipeline import Pipeline
# custom modules
from model.generator import SiameseOnlineGenerator, SupervisedGenerator
from model.model import SiameseTripletModel, SupervisedModel
from core.util import GlobalSettings

tqdm.pandas()

gs = GlobalSettings()

class SiamesePredictor():

    def __init__(self, model: Union[SiameseTripletModel, SupervisedModel, None], 
                 train: Union[pd.DataFrame, str], 
                 test: Union[pd.DataFrame, None] = None, 
                 support_index: Union[list, None] = None, 
                 val_index: Union[list, None] = None) -> None:
        '''
        2nd type in input arguments are for predicting on test set (without ground truth).
        train: dataframe or path to h5 file from export_train_h5(). 
            columns: rel_path, img_name, label, path (optional)
            column to be addded: embedding
        support_index: if None, then use all training data.
        '''
        self._model = model
        if isinstance(train, str):
            train_df = pd.read_hdf(train, key='train_df')
            # add 'path' column
            data_folder = gs.get('data_folder')
            train_df['path'] = train_df['rel_path'].apply(lambda x: os.path.join(data_folder, x))
            # convert category columns back
            cat_col = train_df.filter(like='_category').columns
            for col in cat_col:
                # replace "nan" with None
                train_df[col] = train_df[col].apply(lambda x: None if x == 'nan' else x)
                train_df[col[:-9]] = train_df[col].astype('category')
            train_df = train_df.drop(columns=cat_col)
        else:
            train_df = train.copy() # complete dataframe cached embedding
        train_df['embedding'] = None
        self._train_df = train_df
        self.set_test(test)
        self.support_index = support_index
        self.val_index = val_index
        self._pipeline = {} # map category name to pipeline
        self._tsne = None
        if isinstance(model, SiameseTripletModel):
            self._generator = SiameseOnlineGenerator(df=pd.DataFrame())
        elif isinstance(model, SupervisedModel):
            self._generator = SupervisedGenerator(df=pd.DataFrame())
        elif model is None:
            pass
        else:
            raise ValueError('Model must be SiameseTripletModel or SupervisedModel.')
        return
    
    def set_test(self, test: Union[pd.DataFrame, None] = None) -> None:
        if test is None:
            self._test_df = None
        else:
            self._test_df = test.copy()
            self._test_df['embedding'] = None
            # cache prediction for each category column
            cat_col = self._train_df.select_dtypes(include='category').columns
            for col in cat_col:
                # column with suffix 'pred'
                self._test_df[col + '_pred'] = None
                # probability list, suffix 'prob'
                self._test_df[col + '_prob'] = None
        return
   
    def get_feature(self, file: Union[str, List[str]]) -> np.ndarray:
        if isinstance(file, list):
            imgs = []
            for f in file:
                img = self._generator.load_image(f, center_crop=True)
                imgs.append(img)
            img = np.stack(imgs, axis=0)
        else:
            img = self._generator.load_image(file, center_crop=True)
            img = np.expand_dims(img, axis=0)
        model = self._model
        if isinstance(model, SiameseTripletModel):
            embedding = model.model.predict(img, verbose=1)
        elif isinstance(model, SupervisedModel):
            _, embedding = model.model.predict(img, verbose=1)
        else:
            raise NotImplementedError('Model type not supported.')
        if isinstance(file, str):
            embedding = embedding.squeeze()
        return embedding
    
    def compute_feature(self) -> None:
        if self.support_index is not None:
            support_index = self.support_index
        else:
            support_index = self._train_df.index.tolist()
        support_index = np.array(support_index)
        # find rows with null embedding
        ri = self._train_df.loc[support_index, 'embedding'].isnull().tolist()
        files = self._train_df.loc[support_index[ri], 'path'].tolist()
        if len(files) > 0:
            embeddings = self.get_feature(files)
            # save embedding
            for i, embedding in zip(support_index[ri], embeddings):
                self._train_df.at[i, 'embedding'] = embedding
        if self.val_index is not None:
            val_index = np.array(self.val_index)
            ri = self._test_df.loc[val_index, 'embedding'].isnull().tolist()
            files = self._test_df.loc[val_index[ri], 'path'].tolist()
            if len(files) > 0:
                embeddings = self.get_feature(files)
                for i, embedding in zip(val_index[ri], embeddings):
                    self._test_df.at[i, 'embedding'] = embedding
        return
    
    def set_classifier(self, pipeline: Pipeline,
                       category: str='label') -> None:
        '''
        set classifier for category.
        '''
        # validate category
        if category not in self._train_df.columns or \
            self._train_df[category].dtype.name != 'category':
            raise ValueError(f'Category {category} must be a categorical column in train_df.')
        self._pipeline[category] = pipeline
        return
    
    def fit(self, category: str='label') -> None:
        '''
        fit classifier for category.
        '''
        if category not in self._pipeline:
            raise ValueError(f'Please set classifier for category {category} first.')
        pipeline = self._pipeline[category]
        if self.support_index is not None: 
            support_df = self._train_df.loc[self.support_index]
        else:
            support_df = self._train_df
        # remove rows with null in category column
        support_df = support_df.dropna(subset=[category])
        # check if embedding is computed
        if support_df['embedding'].isnull().sum() > 0:
            raise ValueError('Please compute features first.')
        embedding = support_df['embedding'].values.tolist()
        code = support_df[category].cat.codes.values
        pipeline.fit(embedding, code)
        print(f'Fitted {category} classifier with {len(support_df)} samples.')
        return
    
    def validate(self, category: str='label') -> Tuple[np.ndarray, np.ndarray]:
        '''
        predict test data and cache the result.
        '''
        # validate category
        if category not in self._train_df.columns or \
            self._train_df[category].dtype.name != 'category':
            raise ValueError(f'Category {category} must be a categorical column in train_df.')
        # check key in pipeline
        if category not in self._pipeline:
            raise ValueError(f'Please set classifier for category {category} first.')
        pipeline = self._pipeline[category]
        if self.val_index is None:
            raise ValueError('Please set validation index first.')
        test_df = self._test_df.loc[self.val_index]
        embedding = test_df['embedding'].values.tolist()
        y_true_code = test_df[category].cat.codes.values
        y_pred_code = pipeline.predict(embedding)
        prob = pipeline.predict_proba(embedding)
        # class_code = pipeline.classes_
        # idx = np.argsort(prob, axis=1)[:,-1]
        # y_pred_code = class_code[idx]
        # cache
        self._test_df.loc[self.val_index, category + '_pred'] = y_pred_code
        # convert to list
        # not working
        # prob = [r for r in prob]
        # self._test_df.loc[self.val_index, category + '_prob'] = prob
        for i, idx in enumerate(self.val_index):
            self._test_df.at[idx, category + '_prob'] = prob[i]
        return y_pred_code, y_true_code
    
    def predict(self, file: Union[str, Image.Image], 
                category: str='label') -> Tuple[str, int]:
        '''
        predict a single image (without groundtruth).
        '''
        if category not in self._pipeline:
            raise ValueError(f'Please set classifier for category {category} first.')
        pipeline = self._pipeline[category]
        embedding = self.get_feature(file)
        y_pred_code = pipeline.predict([embedding])
        y_pred_code = y_pred_code[0]
        y_pred_label = self.get_label_from_code(y_pred_code, category=category)
        return y_pred_label, y_pred_code
    
    def predict_and_reference(self, file: Union[str, Image.Image], 
                              category: str='label') -> \
        Tuple[List[str], List[int], list, list]:
        '''
        predict a single image (without groundtruth) and return mapped tsne 
        embedding.
        '''
        if self._tsne is None:
            raise ValueError('Please set tsne estimitor first.')
        if category not in self._pipeline:
            raise ValueError(f'Please set classifier for category {category} first.')
        pipeline = self._pipeline[category]
        # compute embedding
        embedding = self.get_feature(file)
        # probability
        prob = pipeline.predict_proba([embedding]) # (n_samples, n_classes)
        class_code = pipeline.classes_
        # sort class code by decending probability
        idx = np.argsort(prob[0])[::-1]
        y_pred_code = class_code[idx]
        y_pred_code = y_pred_code.tolist()
        y_pred_label = self.get_label_from_code(y_pred_code, category=category)
        prob = prob[0][idx]
        prob = prob.tolist()
        # reference
        embedding = np.array([embedding])
        tsne_embedding_query = self._tsne.transform(embedding)
        tsne_embedding_query = tsne_embedding_query.tolist()[0]
        return y_pred_label, y_pred_code, prob, tsne_embedding_query

    @validate_call
    def get_label_from_code(self, code: Union[List[int], int], 
                            category: str='label') -> Union[List[str], str]:
        map = self._train_df[category].cat.categories
        if isinstance(code, list):
            return map[code].tolist()
        else:
            return map[code]
    
    @validate_call
    def get_code_from_label(self, label: Union[List[str], str], 
                            category: str='label') -> Union[List[int], int]:
        map = dict(enumerate(self._train_df[category].cat.categories))
        reverse_map = {v: k for k, v in map.items()}
        if isinstance(label, list):
            return [reverse_map[l] for l in label]
        else:
            return reverse_map[label]    
    
    def export_test_csv(self, file: str, category: str='label') -> None:
        '''
        export test data to csv without embedding. 
        '''
        # validate category
        if category not in self._test_df.columns or \
            self._test_df[category].dtype.name != 'category':
            raise ValueError(f'Category {category} must be a categorical column in test_df.')
        df = self._test_df.loc[self.val_index]
        # remove embedding, path
        # keep rel_path, img_name
        df = df[['rel_path', 'img_name', category, category + '_pred', category + '_prob']].copy()
        # generate cat code
        df[category + '_code'] = df[category].cat.codes
        # prob
        class_code = self._pipeline[category].classes_
        for i, code in enumerate(class_code):
            df[category + f'_prob_{code}'] = df[category + '_prob'].apply(lambda x: x[i])
        # remove _prob
        df.drop(columns=[category + '_prob'], inplace=True)
        df.to_csv(file)
        return

    def export_train_h5(self, file: str) -> None:
        '''
        export training data excluding embedding. 
        '''
        # remove path
        df = self._train_df.drop(columns=['path', 'embedding'])
        # hd5 doesn't support categorical data in fixed format
        # also nd.array in column is not supported in table format
        # convert categorical columns to string 
        cat_col = df.select_dtypes(include='category').columns
        for col in cat_col:
            # rename column as name_category, assuming original data is string
            df[col + '_category'] = df[col].astype(str)
        df.drop(columns=cat_col, inplace=True)
        df.to_hdf(file, key='train_df', mode='w', format='fixed')
        print(f'Exported training dataframe to {file}')
        return

    def export_tsne_train(self, file: str) -> None: 
        '''
        export t-SNE embedding of training set to csv. 
        '''
        if self._tsne is None:
            raise ValueError('Please set tsne estimator first.')
        if self.support_index is None:
            support_df = self._train_df
        else:
            print('Warning: exporting tsne without full training set')
            support_df = self._train_df.loc[self.support_index]
        X_2d = np.array(self._tsne)
        tsne_df = support_df.drop(columns=['embedding', 'path'])
        tsne_df['tsne_d1'] = X_2d[:, 0]
        tsne_df['tsne_d2'] = X_2d[:, 1]
        tsne_df.to_csv(file, index=True)
        print(f'Exported to {file}')
        return
    
    def compute_tsne(self) -> None:
        '''
        compute tsne embedding for all training data.
        '''
        if self._tsne is not None:
            return
        if self.support_index is None:
            support_df = self._train_df
        else:
            support_df = self._train_df.loc[self.support_index]
        train_embedding = np.vstack(support_df['embedding'].values)
        tsne_estimator = openTSNE.TSNE(n_components=2,
                                    perplexity=30,
                                    metric='euclidean', 
                                    initialization='pca',
                                    n_iter=1000,
                                    n_jobs=-1,
                                    random_state=0)
        tsne_embedding = tsne_estimator.fit(train_embedding)
        self._tsne = tsne_embedding
        return

    def export_tsne(self, file: str) -> None: 
        if self._tsne is None: 
            raise ValueError('Please set tsne estimator first.')
        joblib.dump(self._tsne, file)
        print(f'Exported to {file}')
        return

    def import_tsne(self, file: str) -> None:
        self._tsne = joblib.load(file)
        return

    def export_embedding(self, file: str) -> None: 
        # keep only embedding and rel_path (key)
        train_df = self._train_df[['rel_path', 'embedding']].copy()
        test_df = self._test_df[['rel_path', 'embedding']].copy()
        # set rel_path as index
        train_df.set_index('rel_path', inplace=True)
        test_df.set_index('rel_path', inplace=True)
        train_df.to_hdf(file, key='train_df', mode='w', format='fixed')
        test_df.to_hdf(file, key='test_df', mode='a', format='fixed')
        print(f'Exported embedding to {file}')
        return

    def import_embedding(self, file: str) -> None:
        with pd.HDFStore(file, 'r') as store:
            train_df = store['train_df']
            test_df = store['test_df']
        # parse data and load embedding based on rel_path
        for r_source in train_df.index:
            # find row in train_df with rel_path
            r_target = self._train_df[self._train_df['rel_path'] == r_source].index
            if len(r_target) == 0:
                continue
            self._train_df.at[r_target[0], 'embedding'] = train_df.at[r_source, 'embedding']
        if self._test_df is not None:
            for r_source in test_df.index:
                r_target = self._test_df[self._test_df['rel_path'] == r_source].index
                if len(r_target) == 0:
                    continue
                self._test_df.at[r_target[0], 'embedding'] = test_df.at[r_source, 'embedding']
        print(f'Imported embedding from {file}')
        return

class PredictorManager():

    def __init__(self) -> None:
        self.predictors = {}

    def put(self, run: str, model: str, predictor: SiamesePredictor) -> None:
        key = (run, model)
        if key in self.predictors:
            raise ValueError('Predictor already exists.')
        self.predictors[key] = predictor
        return
    
    def get(self, run: str, model: str) -> SiamesePredictor:
        key = (run, model)
        if key not in self.predictors:
            raise ValueError('Predictor does not exist.')
        return self.predictors[key]
    
    def remove(self, run: str, model: str) -> None:
        key = (run, model)
        if key not in self.predictors:
            raise ValueError('Predictor does not exist.')
        del self.predictors[key]
        return
    
    def has(self, run: str, model: str) -> bool:
        key = (run, model)
        return key in self.predictors
