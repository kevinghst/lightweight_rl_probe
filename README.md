# Light-weight probing of unsupervised representations for Reinforcement Learning 

*Wancong Zhang, Anthony Chen, Vlad Sobol, Yann LeCun, Nicolas Carion*

## Install 
```bash
# PyTorch
export LANG=C.UTF-8
# Install requirements
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Finally, install the project
pip install --user -e .
```

## Prepare Datasets
1. Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install)
2. Download from [DQN Replay Dataset](https://research.google/tools/datasets/dqn-replay/) using the following bash script:

```bash scripts/download_replay_dataset.sh $DATA_DIR```

3. After download, prepare the pretrain and probing datasets (modify ```data_path``` variable on line 8).

```python scripts/prepare_datasets.py```

## Usage:

Pretraining scripts are in scripts/experiments/pretrain, and RL finetuning scripts are in scripts/experiments/finetune. The name of each file is a code for a 
particular pretraining setup. They correspond to the same codes in Table 12 of the paper.

Setup names are represented as {encoder}-{transition model}-{ssl losses}. **M** and **L** refer to ResNet M and ResNet L, **CD** is convolutional model, 
**GD** is deterministic GRU, **GL** is latent GRU, **By** and **Bt** refer to Byol and Barlow, **G** and **I** refer to goal and inverse losses.

Example:

First, pretrain M_GL_Bt0.7I setup on frostbite using seed 1. Assume data lives in ```/home/data```, and checkpoint will be saved in ```checkpoints/my_run```:

```bash scripts/experiments/pretrain/M_GL_Bt0.7I.sh frostbite 1 my_run /home/data```

To perform RL finetuning using the last checkpoint from the above pretrained model, execute:

```bash scripts/experiments/finetune/M_GL_Bt0.7I.sh frostbite 1 my_run```

To perform reward probing on the above pretrained model, execute:

```bash scripts/experiments/probe/M_GL_Bt0.7I.sh frostbite 1 my_run /home/data```

Note that the probing script is almost identical with pretraining script, except for ```eval_only```, ```no_init_eval```, and ```model_load``` flags

To do action probing, set ```algo.probe_task=next_action``` and ```offline.runner.dataloader.ft_ckpt=50```.

## References:
This repo builds on top of [SGI](https://github.com/mila-iqia/SGI) (Schwarzer 2021).
