'''
usage
    uvicorn web.worker:app --reload --port 8000
'''
import asyncio
import logging
from typing_extensions import Annotated
from fastapi import FastAPI, Request, status, Depends, Query
from fastapi.responses import JSONResponse
# custom modules
from web.schema import JobModal
from web.database import get_database
import core.derived
# database
from bson.objectid import ObjectId

# app
app = FastAPI(title="Worker", version="0.1.0")

# logger
logger = logging.getLogger('uvicorn')
logger.setLevel(logging.DEBUG)

# error
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": str(exc)}
    )

# queue
fifo_queue = asyncio.Queue()

# database
db = get_database()
job_collection = db['job']
query_result_collection = db['query_result']

# page
@app.get("/", response_class=JSONResponse)
async def root():
    return {"message": "Worker is running"}

# worker
async def fifo_worker():
    logger.debug("Starting task queue")
    while True:
        queue_job = await fifo_queue.get()
        job_function, *args = queue_job
        logger.debug(f"Got a job: (size of remaining queue: {fifo_queue.qsize()})")
        job_status = await job_function(*args)
        # update job status
        job = args[0]
        if job_status:
            await job_collection.delete_one({"_id": ObjectId(job.id)})
            # await job_collection.update_one(
            #     {"_id": ObjectId(job.id)},
            #     {"$set": {"status": "done"}}
            # )
        else:
            await job_collection.update_one(
                {"_id": ObjectId(job.id)},
                {"$set": {"status": "error"}}
            )
        # query for next job
        await get_next_job()

async def process_job(job: JobModal) -> bool:
    logger.debug(f"Processing job {job.id}")
    data = job.data
    if job.type == "predict":
        await predict_task(data['image_id'])
    else:
        raise ValueError(f"Unknown job type: {job.type}")
    logger.debug(f"Done processing job {job.id}")
    return True

async def get_next_job():
    j = await job_collection.find_one_and_update(
        {"status": "pending"}, 
        {"$set": {"status": "processing"}})
    if j:
        job = JobModal(**j)
        logger.debug(f"Got job {job.id}")
        await fifo_queue.put((process_job, job))

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fifo_worker())
    asyncio.create_task(monitor_result())

async def monitor_result():
    logger.debug("Starting job watcher")
    async with job_collection.watch(
        pipeline=[{"$match": {"operationType": "insert"}}], 
        full_document='updateLookup') as stream:
        async for change in stream:
            job_entry = change['fullDocument']
            logger.debug(f"Change detected: {change}")
            # send job to worker
            job = JobModal(**job_entry)
            await fifo_queue.put((process_job, job))
            
# validate
async def validate_model(run: Annotated[str, Query(title="run", description="Input run name")],
                         model: Annotated[str, Query(title="model", description="Input model name")]):
    run, model, _ = core.derived.validate_model(run, model)
    return {"run": run, "model": model}

# workflow
# 1. job add entry, status = pending
# 2. worker get job, update status = processing
# 3. worker process job, delete job entry, and add result entry

# model
import io
from PIL import Image
from web.model import get_query_image_data
from model.predictor import PredictorManager
from core.derived import create_predictor

pm = PredictorManager()

async def predict_task(image_id: str) -> None:
    logger.debug('enter predict_task')
    results = await simple_predict(image_id, 
                                   {"run": "20240201170939", "model": "weights_e30_final_001"})
    # save to database
    query_result_collection.insert_one({
        'image_id': image_id,
        'reference_label': results['refernece_label'],
        'label': results['label'],
        'probability': results['probability'],
        'tsne': results['tsne']})
    return

async def simple_predict(image_id: str, model_info: Annotated[dict, Depends(validate_model)]):
    '''
    Predict an image with model loading. 
    '''
    # load image
    data = await get_query_image_data(image_id)
    img = Image.open(io.BytesIO(data['content']))
    # load reference label from filename
    refmap = {
        "BA": "basophil",
        "EO": "eosinophil",
        "ERB": "erythroblast",
        "IG": "ig",
        "MMY": "ig (metamyelocyte)",
        "MY": "ig (myelocyte)",
        "PMY": "ig (promyelocyte)",
        "LY": "lymphocyte",
        "MO": "monocyte", 
        "BNE": "neutrophil (band)",
        "SNE": "neutrophil (segmented)",
        "PLATELET": "platelet",
        "BL": "blast",
        "UL": "unidentified"
    }
    # get prefix from before "_"
    prefix = data['filename'].split("_")[0]
    ref_label = refmap.get(prefix, "unknown")
    # load model config
    run = model_info['run']
    model = model_info['model']
    if pm.has(run, model):
        predictor = pm.get(run, model)
    else:
        predictor = create_predictor(run, model)
        pm.put(run, model, predictor)
        logger.debug('predictor created')
    # predict
    y_pred_label, _, prob, tsne_embedding = predictor.predict_and_reference(img)
    return {"image_id": image_id, 
            "refernece_label": ref_label,
            "label": y_pred_label, 
            "probability": prob, 
            "tsne": tsne_embedding}
