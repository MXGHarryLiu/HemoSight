'''
Model evaluation main function
Usage:
    python -m model.val --run 20231118123456 --model best --cfg config.json
Parameters:
    --run: training job ID
    --model: model weights file or strategy name, e.g. 'best' (default), 
        'final', 'weights_e01_best.h5'. 
    --cfg: configuration json file path or json string
'''

import argparse
import json
import os
import re
import itertools
import pandas as pd
import numpy as np
from typing import List
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from modAL.models import ActiveLearner
import modAL.uncertainty
import modAL.density
# custom modules
from model.predictor import SiamesePredictor
from model.loader import Loader, load_loader
from model.model import load_model, SupervisedModel
from model.generator import get_generator
from core.util import GlobalSettings
import core.derived

gs = GlobalSettings()

def main() -> None:
    # parse input
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--run', type=str, 
                        default='', 
                        help='training job ID')
    parser.add_argument('--model', type=str, 
                        default='best', 
                        help='model weights file or strategy name (''best'', ''final'')')
    parser.add_argument('--cfg', type=str, 
                        default='config.json',
                        help='configuration json file path or json string')
    args = parser.parse_args()
    # run
    run, run_folder = core.derived.validate_run(args.run)
    out_folder = os.path.join(gs.get('derived_folder'), run_folder)
    # model
    model = args.model
    if model == 'best':
        weights_file = core.derived.get_default_model_file(run=run, strategy='Best')
        weights_file = os.path.join(out_folder, weights_file)
    elif model == 'final':
        weights_file = core.derived.get_default_model_file(run=run, strategy='Final')
        weights_file = os.path.join(out_folder, weights_file)
    else:
        weights_file = os.path.join(out_folder, model)
    if not os.path.isfile(weights_file):
        raise ValueError(f'Weights file {weights_file} not found.')
    # run cfg
    cfg_file = os.path.join(out_folder, 'config.json')
    with open(cfg_file, 'r') as f:
        run_opt = json.load(f)
    # result folder named as weights file
    model_name = os.path.splitext(os.path.basename(weights_file))[0]
    result_folder = get_next_folder_name(out_folder, model_name)
    os.makedirs(result_folder, exist_ok=False)
    # model cfg
    cfg = args.cfg
    if os.path.isfile(cfg):
        with open(cfg, 'r') as f:
            opt = json.load(f)
    else:
        try:
            opt = json.loads(cfg)
        except Exception as e:
            raise ValueError(f'cfg is not a valid file path or invalid json format: {e}')
    # save config file to output folder
    with open(os.path.join(result_folder, 'config.json'), 'w') as f:
        json.dump(opt, f, indent=4)
    
    # load model
    model_opt = run_opt['model']
    model_opt['out_folder'] = out_folder
    model_opt['model']['weights'] = weights_file
    model_class = load_model(model_opt)
    model = model_class.model

    # load data
    loader_opt = run_opt['loader']
    data_folder = gs.get('data_folder')
    loader_opt['file_list'] = os.path.join(data_folder, loader_opt['file_list'])
    loader = load_loader(loader_opt)
    train_df = loader.train_df
    if loader_opt['kfold'] > 1:
        val_df = loader.val_df
    else:
        val_df = loader.test_df

    # intercept supervised model
    if isinstance(model_class, SupervisedModel):
        generator_opt = run_opt['generator']
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
        # inference validation set
        train_generator, val_generator = get_generator(train_df, val_df, generator_opt)
        y_true_code = val_df[category].cat.codes.values
        y_pred_prob, _ = model.predict(val_generator, verbose=1)
        y_pred_code = np.argmax(y_pred_prob, axis=1)
        classes = train_df[category].cat.categories.tolist()
        class_codes = [i for i in range(len(classes))]
        support_class_codes = train_df[category].cat.codes.tolist() # use all training data
        if n is None:
            file_suffix = f'{category}_all'
        else:
            file_suffix = f'{category}_n{n}'
        cr_file = os.path.join(result_folder, f'classification_report_{file_suffix}.csv')
        export_classification_report(cr_file, y_true_code, y_pred_code, 
                                    class_codes, classes, support_class_codes)
        # tsne
        if opt.get('tsne', False) == True:
            export_supervised_tsne(os.path.join(result_folder, 'tsne2.csv'), 
                                model, train_generator)
        return

    # append morphology categories
    categories = opt['categories']
    morph_categories = [col for col in categories if col != 'label']
    if len(morph_categories) > 0:
        csv_path = os.path.join(data_folder, 'wbcatt', 'pbc_attr_v1.csv')
        train_df, val_df = Loader.load_morph(train_df, val_df, csv_path, morph_categories)

    # inference
    predictor = SiamesePredictor(model=model_class, 
                                 train=train_df, 
                                 test=val_df, 
                                 support_index=None, 
                                 val_index=val_df.index.values) # validate using all samples
    # load embedding from cache
    cache_file = os.path.join(out_folder, f'embedding_{model_name}.h5')
    cached_embedding = False
    if os.path.isfile(cache_file):
        predictor.import_embedding(cache_file)
        cached_embedding = True
    
    # classifier
    classifier_opt = opt['classifier']
    for category in categories:
        predictor.set_classifier(create_classifier(classifier_opt), category)

    # support set    
    support_opt = opt['support']
    is_active = support_opt['active_learning']['mode'] != 'off'
    n_support = support_opt['n']
    sampling = support_opt['sampling']
    if len(n_support) == 0:
        raise ValueError('Support set size n must not be empty.')
    if len(n_support) != len(set(n_support)):
        raise ValueError('Support set size n must not have duplicate values.')
    if is_active:
        if n_support[0] == None:
            raise ValueError('Active learning n_support must not start with None.')
        if None in n_support and n_support[-1] is not None:
            raise ValueError('Active learning n_support None must be at the end.')
        n_support_nonull = [n for n in n_support if n is not None]
        if n_support_nonull != sorted(n_support_nonull):
            raise ValueError('Active learning n_support must be increasing.')
        if len(categories) > 1:
            raise ValueError('Active learning with multiple categories not implemented.')
    n_index = range(len(n_support))
    # create loop variables
    combination = itertools.product(n_index, categories)
    for ni, category in combination:
        # compute minimum of number of samples per class
        value_counts = train_df[category].value_counts()
        min_sample_per_class = value_counts.min()
        n_class = len(value_counts) # number of classes
        n = n_support[ni]
        if sampling == 'stratified' and is_active == False:
            if n is not None and n // n_class >= min_sample_per_class:
                # skip
                continue
            if n is None:
                support_index = train_df.index.tolist()
                print(f'All {len(support_index)} samples selected.')
            else:
                n_per_class = n // n_class
                support_index = Loader.stratified_sample(train_df, category, n_per_class).index.tolist()
                print(f'Stratified sampled {len(support_index)} samples.')
        elif is_active == True:
            if ni == 0:
                # support_index + support_index_pool = total training data
                # n is not None
                active_init = support_opt['active_learning']['initialization']
                n_per_class = n // n_class
                if active_init == 'stratified':
                    support_index = Loader.stratified_sample(train_df, category, n_per_class).index.tolist()
                elif active_init == 'kmeans':
                    # not so good performance
                    embedding = np.vstack(predictor._train_df['embedding'].values.tolist())
                    index = kmeans_active_init(embedding, n_class, n_per_class)
                    support_index = [train_df.index[i] for i in index]
                else:
                    raise NotImplementedError(f'Active learning initialization {active_init} not implemented.')
                # don't use set to preserve order
                support_index_pool = [i for i in train_df.index.tolist() if i not in support_index]
                # learner
                if sampling == 'uncertainty':
                    query_strategy = modAL.uncertainty.uncertainty_sampling
                elif sampling == 'margin':
                    query_strategy = modAL.uncertainty.margin_sampling
                elif sampling == 'entropy':
                    query_strategy = modAL.uncertainty.entropy_sampling
                elif sampling == 'random':
                    def random_sampling(classifier, # BaseEstimator
                                        X_pool, # modALinput
                                        n_instances: int=1, **kwargs):
                        rng = np.random.RandomState(0)
                        return rng.choice(len(X_pool), n_instances, replace=False)
                    query_strategy = random_sampling
                elif sampling == 'euclidean_density':
                    # tried not better than random with euclidean or cosine
                    def euclidean_information_density(classifier, X_pool, 
                                                      n_instances: int=1, **kwargs):
                        euclidean_density = modAL.density.information_density(X_pool, 'euclidean')
                        # return indices of n_instances highest density
                        return np.argsort(euclidean_density)[-n_instances:]
                    query_strategy = euclidean_information_density
                elif sampling == 'disagreement_uncertainty':
                    raise NotImplementedError(f'Active learning with {sampling} sampling not implemented.')
                else:
                    raise NotImplementedError(f'Active learning with {sampling} sampling not implemented.')
                learner = ActiveLearner(estimator=predictor._pipeline[category], 
                                        query_strategy=query_strategy, 
                                        X_training=predictor._train_df['embedding'].loc[support_index].values.tolist(),
                                        y_training=train_df[category].loc[support_index].cat.codes.tolist())
                print(f'Active learning: initial {len(support_index)} samples selected ({active_init}).')
            else:
                if n is None: 
                    # use all samples
                    # n_sample = len(support_index_pool)
                    support_index = train_df.index.tolist()
                    support_index_pool = []
                    print(f'Active learning: all {len(support_index)} samples selected.')
                else:
                    n_sample = int(n - n_support[ni-1])
                    predictor.support_index = support_index_pool
                    predictor.compute_feature()
                    query_idx, _ = learner.query(predictor._train_df['embedding'][support_index_pool].values.tolist(),
                                                    n_instances=n_sample,
                                                    return_metrics=False)
                    support_index_delta = [support_index_pool[i] for i in query_idx]
                    # merge support_index_delta to support_index
                    support_index += support_index_delta
                    support_index_pool = [i for i in support_index_pool if i not in support_index_delta]
                    print(f'Active learning: {len(support_index_delta)} samples selected by {sampling}.')
        # compute feature
        predictor.support_index = support_index
        predictor.compute_feature()
        predictor.fit(category)
        # validate
        y_pred_code, y_true_code = predictor.validate(category)
        classes = train_df[category].cat.categories.tolist()
        class_codes = [i for i in range(len(classes))]
        support_class_codes = train_df[category].loc[support_index].cat.codes.tolist()
        # format file name
        if n is None:
            file_suffix = f'{category}_all'
        else:
            file_suffix = f'{category}_n{n}'
        # compute classification report
        cr_file = os.path.join(result_folder, f'classification_report_{file_suffix}.csv')
        export_classification_report(cr_file, y_true_code, y_pred_code, 
                                    class_codes, classes, support_class_codes)
        # compute confusion matrix
        cm_file = os.path.join(result_folder, f'confusion_matrix_{file_suffix}.csv')
        export_confusion_matrix(cm_file, y_true_code, y_pred_code, 
                                class_codes, classes)
        # export labels
        lbl_file = os.path.join(result_folder, f'labels_{file_suffix}.csv')
        predictor.export_test_csv(lbl_file, category=category)
        print(f'Category {category} n={n} done.')
    
    if cached_embedding == False:
        # save embedding
        predictor.export_embedding(cache_file)

    if is_active == False:
        # save training embeddings
        predictor.export_train_h5(os.path.join(result_folder, 'train_df.h5'))
        
        # openTSNE
        predictor.compute_tsne()
        predictor.export_tsne(os.path.join(result_folder, 'tsne_embedding.joblib'))
        predictor.export_tsne_train(os.path.join(result_folder, 'tsne2.csv'))
    return

def create_classifier(opt: dict) -> Pipeline:
    if opt['name'] == 'SVC':
        pipeline = make_pipeline(StandardScaler(), 
                                SVC(kernel=opt.get('kernel', 'linear'),
                                    C=opt.get('C', 1.0),
                                    decision_function_shape='ovr',
                                    probability=True, 
                                    random_state=0))
    elif opt['name'] == 'XGBoost':
        pipeline = make_pipeline(StandardScaler(), 
                                XGBClassifier(max_depth=6, 
                                            learning_rate=0.2, # default 0.3
                                            n_estimators=100,
                                            random_state=0))
    elif opt['name'] == 'GaussianNB':
        pipeline = make_pipeline(StandardScaler(), 
                                 GaussianNB())
    elif opt['name'] == 'MLP':
        pipeline = make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=tuple(opt['hidden_layer_sizes']),
                                            activation='relu',
                                            solver='adam',
                                            learning_rate_init=0.01,
                                            learning_rate='adaptive',
                                            random_state=0))
    else:
        raise NotImplementedError(f'Classifier {opt["name"]} not implemented.')
    return pipeline

from sklearn.cluster import KMeans

def kmeans_active_init(embedding: np.ndarray, k: int, n_per_k: int=10) -> List[int]:
    '''
    Initialize active learning support set using kmeans
    '''
    # kmean clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embedding)
    # get k cluster centers
    cluster_centers = kmeans.cluster_centers_
    # get n_per_k samples from each cluster near the center
    index = []
    for i in range(k):
        center = cluster_centers[i]
        dist = np.linalg.norm(embedding - center, axis=1)
        idx = np.argsort(dist)[:n_per_k]
        index.extend(idx)
    # assume no duplicates 
    assert len(set(index)) == k * n_per_k
    # it is possible that some classes are not represented
    return index

def get_next_folder_name(root_folder: str, base_name: str) -> str:
    '''
    return the next available folder name (base_name_###) under root_folder
    '''
    existing_folders = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
    matching_folders = [folder for folder in existing_folders if re.match(f"{base_name}_\d+", folder)]

    if not matching_folders:
        return os.path.join(root_folder, f"{base_name}_001")
    existing_numbers = [int(folder[len(base_name)+1:]) for folder in matching_folders]
    next_number = max(existing_numbers) + 1
    next_folder_name = os.path.join(root_folder, f"{base_name}_{str(next_number).zfill(3)}")
    return next_folder_name

def export_classification_report(file: str, y_true_code: np.ndarray, 
                                 y_pred_code: np.ndarray,
                                 class_codes: list,
                                 classes: list,
                                 support_class_codes: list) -> None:
    # remove -1 (null)
    y_true_code_clean = y_true_code[y_true_code != -1]
    y_pred_code_clean = y_pred_code[y_true_code != -1]
    cr = classification_report(y_true_code_clean, y_pred_code_clean, 
                                output_dict=True, 
                                labels=class_codes,
                                target_names=classes)
    support_count = Counter(support_class_codes)
    # rename 'support' to 'count' and add support set size
    for k, v in cr.items():
        if isinstance(cr[k], dict): 
            if 'support' in cr[k]:
                cr[k]['count'] = cr[k].pop('support')
            if k in classes:
                cr[k]['support'] = support_count[classes.index(k)]
            else: 
                cr[k]['support'] = len(support_class_codes)
    cr_df = pd.DataFrame(cr).transpose()
    if 'micro avg' in cr_df.index:
        # move 'micro avg' to the end
        row = cr_df.loc[['micro avg']]
        cr_df.drop(['micro avg'], inplace=True)
        cr_df = pd.concat([cr_df, row])
    elif 'accuracy' in cr_df.index:
        # rename 'accuracy' to 'micro avg'
        acc = cr_df.loc['accuracy']['precision']
        count = cr_df.loc['macro avg']['count']
        row = pd.DataFrame({'precision': [acc], 'recall': [acc], 
                            'f1-score': [acc], 'count': [count], 
                            'support': [len(support_class_codes)]}, 
                        index=['micro avg'])
        cr_df.drop(['accuracy'], inplace=True)
        cr_df = pd.concat([cr_df, row])
    cr_df.to_csv(file)
    return

def export_confusion_matrix(file: str, y_true_code: list,
                            y_pred_code: list, class_codes: list, 
                            classes: list) -> None:
    if -1 in y_true_code:
        classes = ['null'] + classes
        class_codes = [-1] + class_codes
    cm = confusion_matrix(y_true_code, y_pred_code, 
                            labels=class_codes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(file)
    return

import openTSNE

def export_supervised_tsne(file: str, model, train_generator):
    category = 'label'
    train_df = train_generator._df
    train_generator._index = np.arange(len(train_df))
    train_generator._shuffle = False
    y_true_code = train_df[category].cat.codes.values
    y_true_prob, embedding = model.predict(train_generator, verbose=1)
    y_pred_code = np.argmax(y_true_prob, axis=1)
    train_embedding = np.vstack(embedding)
    tsne_estimator = openTSNE.TSNE(n_components=2,
                                    perplexity=30,
                                    metric='euclidean', 
                                    initialization='pca',
                                    n_iter=1000,
                                    n_jobs=-1,
                                    random_state=0)
    tsne_embedding = tsne_estimator.fit(train_embedding)
    # export
    X_2d = np.array(tsne_embedding)
    tsne_df = train_df.drop(columns=['path'])
    tsne_df[category + '_pred'] = y_pred_code
    tsne_df[category + '_code'] = y_true_code
    tsne_df['tsne_d1'] = X_2d[:, 0]
    tsne_df['tsne_d2'] = X_2d[:, 1]
    tsne_df.to_csv(file, index=True)
    print(f'Exported to {file}')
    pass

if __name__ == '__main__':
    main()
