import io
import os
import pandas as pd
from typing import List
from typing_extensions import Annotated
from datetime import datetime, timezone
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.param_functions import Query
# database
from bson.objectid import ObjectId
# custom module
from web.user import get_current_user
from web.schema import QueryImageModel, ProjectModel, JobModal
from web.database import get_database
from core.util import GlobalSettings
import core.derived

# configuration
gs = GlobalSettings()

# router
router = APIRouter()

# database
db = get_database()
query_image_collection = db['query_image']
project_collection = db['project']
query_result_collection = db['query_result']
job_collection = db['job']

# project
@router.post("/project", response_class=JSONResponse,
             response_model=ProjectModel,
             status_code=status.HTTP_201_CREATED,
             response_model_by_alias=False)
async def create_project(user: Annotated[dict, Depends(get_current_user)], 
                         name: Annotated[str, Query(title="name", description="project name")]):
    project_entry = ProjectModel(
        user_id=user.id,
        name=name,
        created_at=datetime.now(timezone.utc),
        status="active"
    )
    new_project = await project_collection.insert_one(
        project_entry.model_dump(by_alias=True, exclude=["id"]))
    project_entry.id = str(new_project.inserted_id)
    return project_entry

@router.get("/project", response_class=JSONResponse, 
            response_model=List[ProjectModel],
            response_model_by_alias=False)
async def get_project(user: dict = Depends(get_current_user)):
    # matcjh id and active
    projects = project_collection.find({"user_id": user.id, "status": "active"})
    result = []
    async for s in projects:
        result.append(ProjectModel(**s))
    return result

@router.delete("/project/{project_id}", response_class=JSONResponse, 
               status_code=status.HTTP_204_NO_CONTENT)
async def remove_project(user: Annotated[dict, Depends(get_current_user)], 
                         project_id: str):
    # update status to inactive
    project = await project_collection.find_one_and_update(
        {"_id": ObjectId(project_id)},
        {"$set": {"status": "inactive"}}
    )
    if project is None:
        raise HTTPException(status_code=400, detail=f"Invalid project id {project_id}")
    return {"id": project_id}

# image
@router.post("/query_image", response_class=JSONResponse,
             status_code=status.HTTP_201_CREATED)
async def create_query_image(image: UploadFile = File(...), 
                             project_id: str = Query(..., title="project_id", description="project id")):
    '''
    upload image and return the image id.
    '''
    contents = await image.read()
    filename = image.filename
    img_entry = QueryImageModel(project_id=project_id,
                            filename=filename,
                            content=contents)
    img = await query_image_collection.insert_one(
        img_entry.model_dump(by_alias=True, exclude=["id"]))
    id = str(img.inserted_id)
    return {"image_id": id}

@router.get("/query_image/{image_id}", response_class=StreamingResponse)
async def get_query_image(image_id: str):
    '''
    retrieve query image by id.
    '''
    data = await get_query_image_data(image_id)
    content = io.BytesIO(data['content'])
    return StreamingResponse(content, media_type='image/jpeg')

async def get_query_image_data(image_id: str) -> dict:
    img_entry = await query_image_collection.find_one(ObjectId(image_id))
    if img_entry is None:
        raise ValueError(f"Invalid image id {image_id}")
    return img_entry

@router.delete("/query_image", response_class=JSONResponse, 
               status_code=status.HTTP_204_NO_CONTENT)
async def remove_query_image(image_id: str):
    '''
    delete query image by id. 
    '''
    result = await query_image_collection.delete_one({"_id": ObjectId(image_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=400, detail=f"Invalid image id {image_id}")
    # drop query result
    result = await query_result_collection.delete_one({"image_id": image_id})
    return {"id": image_id, "deleted_count": result.deleted_count}

@router.get("/query_image_project/{project_id}", response_class=JSONResponse,
            response_model=List[str])
async def get_query_image_project(project_id: str):
    '''
    retrieve all query images (ids) of a project.
    '''
    images = query_image_collection.find({"project_id": project_id})
    result = [str(img["_id"]) async for img in images]
    return result

# results
@router.get("/query_result/{image_id}", response_class=JSONResponse)
async def get_query_result(user: Annotated[dict, Depends(get_current_user)], 
                           image_id: str):
    '''
    retrieve query result by image id.
    '''
    result_entry = await query_result_collection.find_one({"image_id": image_id}, 
                                                          {'_id': 0})
    img_entry = await query_image_collection.find_one({"_id": ObjectId(image_id)})
    if result_entry is not None:
        result = {'tsne': result_entry['tsne'], 
                'label': result_entry['label'], 
                'probability': result_entry['probability'],
                'reference_label': result_entry['reference_label'],
                'status': 'done',
                'filename': img_entry['filename']}
        return result
    # check if image is being processed
    job = await job_collection.find_one({"type": "predict", "data.image_id": image_id})
    if job is not None:
        result = {'status': 'processing', 
                  'filename': img_entry['filename']}
        return result
    # check if image exists
    if img_entry is None:
        result = {'status': 'error'}
    else: 
        result = {'status': 'pending', 
                  'filename': img_entry['filename']}
    return result

@router.get("/tsne_query", response_class=JSONResponse)
async def get_tsne_query(project_id: str):
    '''
    retrieve tsne results of a project. 
    '''
    pipeline = [
        {
            "$match": {
                "project_id": project_id
            }
        },
        {
            "$addFields": {
                "image_id": { "$toString": "$_id" },
                "filename": "$filename"
            }
        },
        {
            "$lookup": {
                "from": "query_result",
                "localField": "image_id",
                "foreignField": "image_id",
                "as": "results"
            }
        },
        {
            "$unwind": "$results"
        },
        {
            "$project": {
                "_id": 0,
                "image_id": 1,
                "filename": 1,
                "label": "$results.label",
                "tsne": "$results.tsne"
            }
        }
    ]
    results = query_image_collection.aggregate(pipeline)
    results = await results.to_list(length=None)
    # convert to dataframe
    df = pd.DataFrame(results)
    data_df = pd.DataFrame()
    data_df['label'] = df['label'].apply(lambda x: x[0])
    data_df['image'] = df['image_id'].apply(lambda x: f"/api/query_image/{x}")
    data_df['tsne_d1'] = df['tsne'].apply(lambda x: x[0])
    data_df['tsne_d2'] = df['tsne'].apply(lambda x: x[1])
    data_df['img_name'] = df['filename']
    data_df['query'] = True
    # append tsne training data from get_tsne
    run = '20240201170939'
    model = 'weights_e30_final_001'
    data_path = os.path.join(gs.get('derived_folder'), run, model, 'tsne2.csv')
    # obtain dataset name
    cfg = core.derived.get_run_config(run)
    dataset = cfg['loader']['file_list']
    # load data
    df = pd.read_csv(data_path, index_col=0)
    ref_df = df[['tsne_d1', 'tsne_d2', 'label', 'img_name']].copy()
    ref_df['image'] = ref_df.index.map(lambda x: f"/api/image_by_id?dataset={dataset}&id={x}")
    ref_df['query'] = False
    # merge vertically
    data_df = pd.concat([ref_df, data_df], ignore_index=True, axis=0)
    return data_df.to_dict(orient='records')

async def get_query_result_insert_stream():
    return query_result_collection.watch(
        pipeline=[{'$match': {'operationType': 'insert'}}], 
        full_document='updateLookup')

@router.get("/query_knn", response_class=JSONResponse)
async def get_query_knn(image_id: str): 
    '''
    compute knn of query image. 
    '''
    # obtain project_id
    img_entry = await get_query_image_data(image_id)
    project_id = img_entry['project_id']
    # query data
    pipeline = [
        {
            "$match": {
                "project_id": project_id
            }
        },
        {
            "$addFields": {
                "image_id": { "$toString": "$_id" },
                "filename": "$filename"
            }
        },
        {
            "$lookup": {
                "from": "query_result",
                "localField": "image_id",
                "foreignField": "image_id",
                "as": "results"
            }
        },
        {
            "$unwind": "$results"
        },
        {
            "$project": {
                "_id": 0,
                "image_id": 1,
                "filename": 1,
                "label": "$results.label",
                "tsne": "$results.tsne"
            }
        }
    ]
    data_df = pd.DataFrame()
    results = query_image_collection.aggregate(pipeline)
    results = await results.to_list(length=None)
    # convert to dataframe
    df = pd.DataFrame(results)
    data_df = pd.DataFrame()
    data_df['image_id'] = df['image_id']
    data_df['label'] = df['label'].apply(lambda x: x[0])
    data_df['image'] = df['image_id'].apply(lambda x: f"/api/query_image/{x}")
    data_df['tsne_d1'] = df['tsne'].apply(lambda x: x[0])
    data_df['tsne_d2'] = df['tsne'].apply(lambda x: x[1])
    data_df['img_name'] = df['filename']
    data_df['type'] = "query"
    # append tsne training data from get_tsne
    run = '20240201170939'
    model = 'weights_e30_final_001'
    data_path = os.path.join(gs.get('derived_folder'), run, model, 'tsne2.csv')
    # obtain dataset name
    cfg = core.derived.get_run_config(run)
    dataset = cfg['loader']['file_list']
    # load data
    df = pd.read_csv(data_path, index_col=0)
    ref_df = df[['tsne_d1', 'tsne_d2', 'label', 'img_name']].copy()
    ref_df['image'] = ref_df.index.map(lambda x: f"/api/image_by_id?dataset={dataset}&id={x}")
    ref_df['type'] = "reference"
    ref_df['image_id'] = ref_df.index
    # merge vertically
    data_df = pd.concat([ref_df, data_df], ignore_index=True, axis=0)
    # compute knn around query image
    query_tsne = data_df.query(f"image_id == '{image_id}'")[['tsne_d1', 'tsne_d2']].values[0]
    def euclidean_distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    data_df['distance'] = data_df.apply(lambda x: euclidean_distance(x['tsne_d1'], x['tsne_d2'], query_tsne[0], query_tsne[1]), axis=1)
    # sort by distance
    data_df = data_df.sort_values(by='distance')
    # set first row type
    data_df.iloc[0, data_df.columns.get_loc('type')] = "current"
    # take top 10
    data_df = data_df.head(10 + 1)
    # reverse order
    data_df = data_df.iloc[::-1]
    # return top 10
    return data_df.to_dict(orient='records')

# job
# API for testing only
@router.post("/job", response_class=JSONResponse, 
             response_model=JobModal)
async def put_job(image_id: str):
    '''
    Put a job. 
    '''
    # check if image_id exists
    await get_query_image_data(image_id)
    # create a job
    job_entry = JobModal(type="predict",
                        data={"image_id": image_id},
                        status="pending", 
                        created_at=datetime.now(timezone.utc), 
                        updated_at=datetime.now(timezone.utc))
    # save to database
    new_job = await job_collection.insert_one(
        job_entry.model_dump(by_alias=True, exclude=["id"]))
    job_entry.id = str(new_job.inserted_id)
    return job_entry
