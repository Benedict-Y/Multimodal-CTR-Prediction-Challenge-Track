# Multimodal CTR Prediction Solution fron Team Concave Opt

## Task1: Multimodal Item Embedding

#### 1.Qwen2.5-vl-7B to get emb

using the second-to-last hidden layer with mean pooling to obtain embeddings

#### 2.Using PCA to reduce the dimensionality of embeddings

#### 3.Using the default settings of the DIN recommendation model

Hyperparameter scheme:  
embedding_regularizer: 0  
net_regularizer: 0  
net_dropout: 0.11  
learning_rate: 2.e-3  
batch_size: 7168  

### Run

#### 1.Data Preparation

Download the datasets at: https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR

#### 2.Model Download
Because the size of LLM and the bad internet connection, please download model folder and put it at main dir  
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
#### 3.Run code

1.Running all code  

```bash
 ./run.sh
```  
run.sh will install libraries from requirements.txt and transformers-main(auto unzip)  
notice!!!: transformers library will install from local(transformers-main.zip) because Qwen2.5-vl-7B depends on some new features  

2.Only part1: get embeddings and PCA  

```bash
python inference.py

``` 
```bash
python PCA.py
```  

3.Only part2: train DIN and predict  

```bash
python gen_item_info.py
```  

```bash
python run_param_tuner.py --config config/DIN_microlens_mmctr_tuner_config_01.yaml --gpu 0
```  

```bash
python prediction.py --config config/DIN_microlens_mmctr_tuner_config_01 --expid DIN_MicroLens_1M_x1_xxx --gpu 0
```  

## Task2: Multimodal CTR Prediction

Our model is roughly based on **Transformer** and **FinalMLP**. 
We also reference https://github.com/pinskyrobin/WWW2025_MMCTR#
## Environment

We run the experiments on a customized 4090 GPU server with 24GB memory from [AutoDL](https://www.autodl.com/).

Requirements:

- fuxictr==2.3.7
- numpy==1.26.4
- pandas==2.2.3
- scikit_learn==1.4.0
- torch==1.13.1+cu117

Environment setup:

```bash
conda create -n fuxictr_momo python==3.9
pip install -r requirements.txt
source activate fuxictr_momo
```

## How to Run

### One-click run

```bash
sh ./run.sh
```

This script will run the whole pipeline, including model training and prediction.

### Run step by step

1. Train the model on train and validation sets:

    ```bash
    python run_expid.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 0
    ```
    

2. Make predictions on the test set:

    ```bash
    python prediction.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 0
    ```

3. Submission result on [the leaderboard](https://www.codabench.org/competitions/5372/#/results-tab).

