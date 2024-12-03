'''
usage
    uvicorn web.main:app --reload --port 8000
'''
import os
import re
import json
import logging
import asyncio
import pandas as pd
from typing import Dict
from typing_extensions import Annotated
from fastapi import FastAPI, Request, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.param_functions import Query
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
# custom module
import web.data as data
from core.util import GlobalSettings
import core.derived
# route
from web.model import router as model_router
from web.model import get_query_image_data, put_job, get_query_result_insert_stream
from web.user import router as user_router

# configuration
gs = GlobalSettings()
TEMPLATE_FOLDER = "web/templates"
VISUAL_FOLDER = "web/visual"

# app
app = FastAPI(title="Explorer", version="0.1.0")
templates = Jinja2Templates(directory=TEMPLATE_FOLDER)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(model_router, prefix="")
app.include_router(user_router, prefix="")

# error
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": str(exc)}
    )

# logger
logger = logging.getLogger('uvicorn')
logger.setLevel(logging.DEBUG)

# page
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("base.html", 
        {"request": request})

# validate 
async def validate_dataset(dataset: Annotated[str, Query(title="dataset", description="Input dataset name")]):
    datasets = await get_dataset()
    if dataset not in datasets:
        raise HTTPException(status_code=400, detail=f"Invalid dataset {dataset}")
    return dataset

async def validate_run(run: Annotated[str, Query(title="run", description="Input run name")]):
    run, _ = core.derived.validate_run(run)
    return run

async def validate_model(run: Annotated[str, Query(title="run", description="Input run name")],
                         model: Annotated[str, Query(title="model", description="Input model name")]):
    run, model, _ = core.derived.validate_model(run, model)
    return {"run": run, "model": model}

async def validate_runs(runs: Annotated[list, Query(title="runs", description="Input run names")]):
    valid_runs = await get_run()
    for run in runs:
        if run not in valid_runs:
            raise HTTPException(status_code=400, detail=f"Invalid run {run}")
    # check if runs are unique
    if len(set(runs)) != len(runs):
        raise HTTPException(status_code=400, detail=f"Runs are not unique")
    return runs

# plot
@app.get("/vega", response_class=JSONResponse)
async def get_vega(name: Annotated[str, Query(title="name", description="Input plot name")]):
    '''
    return Vega plot definition json.
    '''
    path = os.path.join(VISUAL_FOLDER, name + '.json')
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"Invalid plot name {name}")
    with open(path, 'r') as f:
        plot = json.load(f)
    return plot

# parameters
@app.get("/dataset", response_class=JSONResponse)
async def get_dataset():
    '''
    return a list of dataset (csv file names) under data_folder, e.g. ['dataset.csv']. 
    '''
    dataset = []
    for file in os.listdir(gs.get('data_folder')):
        if file.endswith(".csv") and not file.startswith("._"):
            dataset.append(file)
    return dataset

@app.get("/run", response_class=JSONResponse, 
         summary=core.derived.get_run.__doc__)
async def get_run():
    runs, _ = core.derived.get_run()
    return runs

@app.get("/model", response_class=JSONResponse, 
         summary=core.derived.get_model.__doc__)
async def get_model(run: str):
    return core.derived.get_model(run)

# data
@app.get("/label_data", response_class=JSONResponse)
async def get_label_data(dataset: Annotated[str, Depends(validate_dataset)]):
    '''
    return a list of label data.
    '''
    data_path = os.path.join(gs.get('data_folder'), dataset)
    df = data.load_dataset(data_path)
    data_df = df[['label', 'is_public']].copy() # 'img_name'
    return data_df.to_dict(orient='records')

@app.get("/run_config", response_class=JSONResponse, 
            summary=core.derived.get_run_config.__doc__)
async def get_run_config(run: str):
    return core.derived.get_run_config(run)

@app.get("/accuracy", response_class=JSONResponse)
async def get_accuracy(model_info: Annotated[dict, Depends(validate_model)]):
    '''
    return a list of accuracy. 
    '''
    run = model_info['run']
    model = model_info['model']
    _, _, model_path = core.derived.validate_model(run, model)
    category = 'label'
    data_path = os.path.join(gs.get('derived_folder'), model_path)
    # search file matching classification_report_{category}_{all|n\d+}.csv
    files = []
    for file in os.listdir(data_path):
        if re.match(r'classification_report_{}_all.csv'.format(category), file):
            files.append((os.path.join(data_path, file), 0))
        elif re.match(r'classification_report_{}_n\d+.csv'.format(category), file):
            files.append((os.path.join(data_path, file), int(re.findall(r'\d+', file)[0])))
    # sort tuple by n
    files.sort(key=lambda x: x[1])
    metric = 'f1-score' # TODO: add metric selection
    pool = 'micro avg'
    metrics = ['precision', 'recall', 'f1-score']
    pools = ['micro avg', 'macro avg', 'weighted avg']
    if metric not in metrics:
        raise HTTPException(status_code=400, detail=f"Invalid metric {metric}")
    if pool not in pools:
        raise HTTPException(status_code=400, detail=f"Invalid pool {pool}")
    description = f"{metric} of {pool}"
    y = []
    x = []
    for file, n in files:
        x.append(n)
        df = pd.read_csv(file, index_col=0)
        m = df.loc[pool][metric]
        y.append(m)
    data = []
    for i in range(len(x)):
        data.append({"label_count": x[i], "accuracy": y[i]})
    return data

@app.get("/train_log", response_class=JSONResponse)
async def get_train_log(run: Annotated[str, Depends(validate_run)]):
    '''
    return a list of train log. 
    '''
    _, run_folder = core.derived.validate_run(run)
    data_path = os.path.join(gs.get('derived_folder'), run_folder, 'training.log')
    df = pd.read_csv(data_path, header=0)
    data = df.to_dict(orient='records')
    return data

@app.get("/loss", response_class=JSONResponse)
async def get_loss(run: Annotated[dict, Depends(validate_run)]):
    '''
    return a list of loss. 
    '''
    _, run_folder = core.derived.validate_run(run)
    data_path = os.path.join(gs.get('derived_folder'), run_folder, 'training.log')
    df = pd.read_csv(data_path, header=0)
    df['epoch'] = df["epoch"] + 1 # start from 1
    # convert {"loss": [1,2], "val_loss": [3,4]} to [{"stage": "train", "epoch": 1, "loss": 3},...
    data = []
    for i, row in df.iterrows():
        data.append({"stage": "train", "epoch": row['epoch'], "loss": row['loss']})
        data.append({"stage": "validate", "epoch": row['epoch'], "loss": row['val_loss']})
    return data

@app.get("/tsne", response_class=JSONResponse)
async def get_tsne(model_info: Annotated[dict, Depends(validate_model)]):
    '''
    return a list of tsne. 
    '''
    run = model_info['run']
    model = model_info['model']
    run, _, model_path = core.derived.validate_model(run, model)
    data_path = os.path.join(gs.get('derived_folder'), model_path, 'tsne2.csv')
    # obtain dataset name
    cfg = core.derived.get_run_config(run)
    dataset = cfg['loader']['file_list']
    # load data
    df = pd.read_csv(data_path, index_col=0)
    data_df = df[['tsne_d1', 'tsne_d2', 'label', 'img_name']].copy()
    data_df['image'] = data_df.index.map(lambda x: f"/api/image_by_id?dataset={dataset}&id={x}")
    return data_df.to_dict(orient='records')

@app.get("/confusion_matrix", response_class=JSONResponse)
async def get_confusion_matrix(model_info: Annotated[dict, Depends(validate_model)]):
    '''
    return confusion matrix data. 
    '''
    run = model_info['run']
    model = model_info['model']
    _, _, model_path = core.derived.validate_model(run, model)
    category = 'label'
    data_path = os.path.join(gs.get('derived_folder'), model_path, 
                             f'confusion_matrix_{category}_all.csv')
    df = pd.read_csv(data_path, index_col=0)
    # add normalized value
    df_norm = df.div(df.sum(axis=1), axis=0)
    # convert to {"y_true": ... "y_pred": ... "count": ...}
    data = []
    for i, row in df.iterrows():
        for j, count in row.items():
            data.append({"y_true": i, "y_pred": j, "count": count, "norm": df_norm.loc[i, j]})
    return data

# raw data
@app.get("/image_by_id", response_class=StreamingResponse)
async def get_image_by_id(dataset: Annotated[str, Depends(validate_dataset)],
                          id: Annotated[int, Query(title="id", description="Input image id")]):
    data_path = os.path.join(gs.get('data_folder'), dataset)
    df = data.load_dataset(data_path)
    try:
        file_path = os.path.join(gs.get('data_folder'), df.iloc[id]['rel_path'])
    except IndexError:
        raise HTTPException(status_code=400, detail=f"Invalid id {id}")
    return StreamingResponse(open(file_path, mode='rb'), media_type='image/jpeg')

# model
# workflow
# 1. user submit image (POST@/query_image) -> image id
# 2. trigger predict task (POST@"predict") -> task submitted
# 3. task finishes and socket return results (label and id)
# 4. trigger image display (GET@/query_image) -> image stream

class WebSocketData(BaseModel): 
    command: str
    argument: dict

class WebSocketManager: 
    def __init__(self): 
        # websocket to project_id
        self._websockets: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        # create websocket id
        # socket_id = str(uuid.uuid4())
        self._websockets[websocket] = ""
        logger.debug('websocket connected')
        return ""
    
    async def subscribe(self, websocket: WebSocket, project_id: str):
        self._websockets[websocket] = project_id
        logger.debug(f'websocket subscribed to project_id {project_id}')
        return
    
    async def disconnect(self, websocket: WebSocket):
        self._websockets.pop(websocket)
        return

    async def send(self, project_id: str, message: dict):
        # broadcast to all websockets of the project
        for websocket, p_id in self._websockets.items():
            if p_id == project_id:
                await websocket.send_json(message)
                logger.debug('websocket sent')
        return

wsm = WebSocketManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await wsm.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            data = WebSocketData(**data)
            if data.command == "subscribe":
                await wsm.subscribe(websocket, data.argument['project_id'])
                logger.debug('websocket subscribed')
            elif data.command == "disconnect":
                await websocket.close()
                await wsm.disconnect(websocket)
                logger.debug('websocket disconnected active')
                break
            elif data.command == "echo":
                logger.debug('websocket echo {}'.format(data.argument['message']))
            else:
                logger.debug(f"Unknown command: {data.command}")
    except WebSocketDisconnect:
        await wsm.disconnect(websocket)
        logger.debug('websocket disconnected passive')

@app.get("/predict", response_class=JSONResponse)
async def predict(image_id: Annotated[str, Query(description="Input image id")]):
    '''
    Predict an image. 
    '''
    logger.debug("predict task added")
    # submit a job
    await put_job(image_id)
    return {"message": "predict task added"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_result())

async def monitor_result():
    # subscribe to new result
    logger.debug("Starting result watcher")
    async with await get_query_result_insert_stream() as stream:
        async for change in stream:
            result_entry = change['fullDocument']
            logger.debug(f"Change detected: {change}")
            # update job status
            image_id = result_entry['image_id']
            await send_update_to_ws(image_id)

async def send_update_to_ws(image_id):
    # get project_id
    image_entry = await get_query_image_data(image_id)
    project_id = image_entry['project_id']
    # send update to websocket
    logger.debug(f'predict task done for image {image_id}')
    await wsm.send(project_id, {"command": "predict", 
                                "argument": {"image_id": image_id}})
    return
