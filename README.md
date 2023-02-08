# F1TENTH Imitation Learning

This repository contains code associated with [A Benchmark Comparison of Imitation Learning-based Control Policies for Autonomous Racing
](https://arxiv.org/abs/2209.15073)

## Quickstart
Clone this repository
```bash
git clone https://github.com/mlab-upenn/f1tenth_il.git
```

Navigate to the root directory of this project
```bash
cd f1tenth_il
```

Create a new conda environment with Python 3.8
```bash
conda create -n f1tenth python=3.8
```

Activate the environment
```bash
conda activate f1tenth
```

Install pip
```bash
conda install pip  
```

Install the dependencies for F1TENTH gym.
```bash
pip install -e .
```

Install other dependencies
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install pandas==1.4.4 pyglet==1.5.11
```

## Usage
Navigate to the imitation learning folder for all of the following.
```bash
cd imitation_learning
```

### Collect Expert Trajectories
Run experts on various maps for various vgain_scales (scales top speed) with one file.
```bash
python expert_inference.py 
```

### Training and Inference for Many Imitation Policies
Train many imitation policies with both unique experts and mixture of experts.
```bash
nohup python -u train_many_policies.py --algorithm=bc --training_config=configs/mixed_seed_0.yaml > logs/mixed_seed_0.out &
nohup python -u train_many_policies.py --algorithm=bc --training_config=configs/mixed_seed_1.yaml > logs/mixed_seed_1.out &
nohup python -u train_many_policies.py --algorithm=bc --training_config=configs/mixed_seed_2.yaml > logs/mixed_seed_2.out &

nohup python -u train_many_policies.py --algorithm=bc --training_config=configs/unique_slow.yaml > logs/unique_slow.out &
nohup python -u train_many_policies.py --algorithm=bc --training_config=configs/unique_normal.yaml > logs/unique_normal.out &
nohup python -u train_many_policies.py --algorithm=bc --training_config=configs/unique_fast.yaml > logs/unique_fast.out &
```

Collect trajectories for all of the above with one file.
```bash
python inference_for_many_policies.py 
```

### Training
Execute the training script
```bash
python train.py --algorithm=<algorithm name> --training_config=<yaml file location>
```

Example:
```bash
python train.py --algorithm=dagger --training_config=il_config.yaml
```


### Inference
Execute the inference script
```bash
python inference.py --training_config=<yaml file location> --model_path=<model path>
```

Example:
```bash
python inference.py --training_config=il_config.yaml --model_path=logs/HGDAgger_model.pkl
```
