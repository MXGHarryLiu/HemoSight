# HemoSight: Adaptive and Efficient Leukocyte Classification

### TL;DR:

This is the repository for the study published in this [paper](https://openreview.net/forum?id=xkgKn92AGp). This paper presents a self-supervised model for classifying white blood cells in peripheral blood smears, achieving high accuracy (F1: 96.2%) while generalizing to diverse label sets. The lightweight EfficientNetV2-B0-based approach enhances label efficiency with active learning and is available as the HemoSight web app to streamline clinical workflows.

## Data preparation and model training
### Organization
General flow of data pipeline: 
1. `curate.py`: curate raw data folders to create data reference csv. 
2. `loader.py`: Load data reference csv and perform train, validation, and test split. 
3. `train.py`, `generator.py`: Train model; data augmentation generated from generator. 
4. `val.py`, `predictor.py`: Validate trained model; predictor performs classification on embeddings. 

Folders: 
* `src` stores python source code. 
* `frontend` stores frontend source code. 
* `data` is used to store raw data.
* `derived` is used to store results.
* `mongodb` is used by the MongoDB database. 
Three folders (`data`, `derived`, `src`) are needed for file I/O, and should be mounted to the container. 

Configuration files: 
* Path configuration file (`/src/core/settings.json`):  
This file is loaded by `util.GlobalSettings`. `settings.json` is used by default but can be overwritten by `settings_{systemname}.json` where `{systemname}` is the computer name such as `ThinkPad`. This enables the same repository code base to be cloned and run on multiple environments. 
* Job configuration file: 
Some examples are located at `/src/config_*.json`. See below for their usage. 

### Testing on Local Computer (Not Recommended)
1. Create environment `conda create -n HemoSight python=3.10` and initialize the environment. 
2. Install packages listed in the dockerfile. 
3. You may also need to install `ipykernel` for editing ipynb files. 

### Run on Docker
`Dockerfile.gpu` is the GPU version. `Dockerfile.cpu` is the CPU version.

1. Build `docker build -t hematology:v1 -f Dockerfile.cpu .`
2. Run the container with above three folder mounted `docker run -it --gpus all --rm -v "$(pwd)/src:/src" -v "$(pwd)/derived:/derived" -v "D:/Drive/Data/Hematology:/data" hematology:v1`
3. After the container is running, execute the following in the container. 
    - Training `python -m model.train --cfg config.json`
    - Validation `python -m model.val --run 20231208192208`

## Model deployment
### Testing on Local Computer (Not Recommended)
You may need to install [node.js 20.11](https://nodejs.org/en/blog/release/v20.11.0), [MongoDB 7.0](https://www.mongodb.com/docs/v7.0/installation/). 

### Installation
1. Build `docker-compose -f docker-compose.dev.yaml up --build`
2. Access `http://localhost:4002` for index page. 

## Citation

If you find this repository helpful in your research or work, please consider citing our paper:

```
@inproceedings{liu2024adaptive,
  title={Adaptive self-supervised learning of morphological landscape for leukocytes classification in peripheral blood smears},
  author={Liu, Z., Castillo, S. P., Han, X., Sun, X., Hu, Z., and Yuan, Y.},
  booktitle={Proceedings of the IEEE-EMBS International Conference on Biomedical and Health Informatics},
  year={2024},
  url={https://openreview.net/forum?id=xkgKn92AGp}
}
```

Feel free to contact us for further information or questions related to the paper and this repository. 

Yuan Lab @ MD Anderson Cancer Center