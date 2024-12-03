import os
import re
import json
from typing import List, Literal, Tuple
from pydantic import validate_call
from core.util import GlobalSettings
from model.predictor import SiamesePredictor
from model.train import load_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import jmespath

gs = GlobalSettings()

# validate
def validate_run(run: str) -> Tuple[str, str]:
    '''
    run: run id, e.g. '20231207152440'
    run_folder: run folder name, e.g. '20231207152440_notes'
    '''
    runs, run_folders = get_run()
    if run not in runs:
        raise ValueError(f'Run {run} not found.')
    run_folder = run_folders[runs.index(run)]
    return run, run_folder

def validate_model(run: str, model: str) -> Tuple[str, str, str]:
    '''
    run: run id, e.g. '20231207152440'
    model: model folder name, e.g. 'weights_e02_best_001'
    model_path: model folder path, e.g. '20231207152440_notes/weights_e02_best_001'
    '''
    run, run_folder = validate_run(run)
    models = get_model(run=run)
    if model not in models:
        raise ValueError(f'Model {model} not found under run {run}.')
    model_path = os.path.join(run_folder, model)
    return run, model, model_path

# data
@validate_call
def get_run() -> Tuple[List[str], List[str]]:
    '''
    return a list of experiment run id and folder name. 
    Run ID e.g. '20231207152440'. 
    Run folder name can append with notes separated by one underscore, 
    e.g. '20231207152440_notes'.
    '''
    runs = []
    run_folders = []
    derived_folder = gs.get('derived_folder')
    for folder in os.listdir(derived_folder):
        if os.path.isdir(os.path.join(derived_folder, folder)):
            if re.match(r'^\d{14}$', folder):
                run_id = folder
                runs.append(run_id)
                run_folders.append(folder)
            elif re.match(r'^\d{14}(_\w+)*', folder):
                run_id = folder.split('_')[0]
                runs.append(run_id)
                run_folders.append(folder)
    # sort both by run id
    runs, run_folders = zip(*sorted(zip(runs, run_folders)))
    return runs, run_folders

@validate_call
def get_model(run: str) -> List[str]: 
    '''
    return a list of model folder names under an experiment run folder, e.g. 
    ['weights_e02_best_001']. 
    '''
    _, run_folder = validate_run(run)
    model_folders = []
    run_folder = os.path.join(gs.get('derived_folder'), run_folder)
    for folder in os.listdir(run_folder):
        if os.path.isdir(os.path.join(run_folder, folder)):
            model_folders.append(folder)
    return model_folders

@validate_call
def get_default_model_file(run: str, 
                           strategy: Literal['Best', 'Final'] = 'Best') -> str:
    '''
    return the default model file name under an experiment run folder, e.g.
    'weights_e02_best.h5'. 
    '''
    _, run_folder = validate_run(run)
    model_files = []
    run_folder = os.path.join(gs.get('derived_folder'), run_folder)
    for file in os.listdir(run_folder):
        if file.endswith('.h5') and os.path.isfile(os.path.join(run_folder, file)):
            model_files.append(file)
    if len(model_files) == 0:
        raise ValueError(f'No model found under run {run}.')
    if strategy == 'Best':
        model_files = [x for x in model_files if re.match(r'weights_e\d+_best', x)]
        model_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        model_file = model_files[-1]
    else: # strategy == 'Final'
        model_files = [x for x in model_files if re.match(r'weights_e\d+_final', x)]
        model_file = model_files[0]
    return model_file

@validate_call
def get_default_model(run: str, 
                      strategy: Literal['Best', 'Final'] = 'Best') -> str:
    '''
    return the default model folder name under an experiment run folder, e.g.
    'weights_e02_best_001'. 
    '''
    model = get_model(run=run)
    if len(model) == 0:
        raise ValueError(f'No model found under run {run}.')
    if strategy == 'Best':
        model = [x for x in model if re.match(r'weights_e\d+_best', x)]
        model.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        if len(model) == 0:
            raise ValueError(f'No best model found under run {run}.')
        model = model[-1]
    else: # strategy == 'Final'
        model = [x for x in model if re.match(r'weights_e\d+_final', x)]
        if len(model) == 0:
            raise ValueError(f'No final model found under run {run}.')
        if len(model) != 1:
            raise ValueError(f'Exist more than one final model under run {run}.')
        model = model[0]
    return model

@validate_call
def get_run_config(run: str) -> dict:
    '''
    return run configuration from config.json. 
    '''
    _, run_folder = validate_run(run)
    data_path = os.path.join(gs.get('derived_folder'), run_folder, 'config.json')
    with open(data_path, 'r') as f:
        cfg = json.load(f)
    return cfg

@validate_call
def get_model_config(run: str, model: str) -> dict:
    '''
    return model configuration from config.json. 
    '''
    _, _, model_path = validate_model(run=run, model=model)
    data_path = os.path.join(gs.get('derived_folder'), model_path, 'config.json')
    with open(data_path, 'r') as f:
        cfg = json.load(f)
    return cfg

@validate_call
def query_run(query: str) -> List[str]:
    '''
    query experiment run folders based on parameters. 
    query: e.g. "model.optimizer.name =='Adam'"
    '''
    runs, _ = get_run()
    match_runs = []
    for run in runs:
        cfg = get_run_config(run)
        r = jmespath.search(query, cfg)
        if r:
            match_runs.append(run)
    return match_runs

# model
@validate_call
def create_predictor(run: str, model: str) -> object: 
    '''
    run: run name, e.g. '20231207152440'
    model: model folder name, e.g. 'weights_e02_best_001'
    '''
    # load model config
    opt = get_run_config(run)
    _, run_folder = validate_run(run)
    run_folder = os.path.join(gs.get('derived_folder'), run_folder)
    # append model weights file
    # remove _000 appendix
    model_name = model[:-4]
    weight_file = model_name + '.h5'
    model_opt = opt['model']
    model_opt['model']['weights'] = os.path.join(run_folder, weight_file)
    # load model
    model_class = load_model(model_opt)
    # load predictor
    h5_file = os.path.join(run_folder, model, 'train_df.h5')
    predictor = SiamesePredictor(model=model_class, 
                                 train=h5_file)
    # load embedding
    cache_file = os.path.join(run_folder, f'embedding_{model_name}.h5')
    if os.path.isfile(cache_file):
        predictor.import_embedding(cache_file)
    pipeline = make_pipeline(StandardScaler(), 
                             SVC(kernel='linear', 
                                 C=1,
                                 decision_function_shape='ovr',
                                 probability=True, 
                                 random_state=0))
    predictor.set_classifier(pipeline, category='label')
    predictor.fit(category='label')
    # load tsne estimator
    tsne_file = os.path.join(run_folder, model, 'tsne_embedding.joblib')
    if os.path.isfile(tsne_file):
        # load cached tsne embedding
        predictor.import_tsne(tsne_file)
    else:
        predictor.compute_tsne()
        predictor.export_tsne(tsne_file)
    return predictor
