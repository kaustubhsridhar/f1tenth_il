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
### Training
Navigate to the imitation learning folder
```bash
cd imitation_learning
```

Execute the training script
```bash
python train.py --algorithm=<algorithm name> --training_config=<yaml file location>
```

Example:
```bash
python train.py --algorithm=dagger --training_config=il_config.yaml
```


### Inference
Navigate to the imitation learning folder
```bash
cd imitation_learning
```

Execute the inference script
```bash
python inference.py --training_config=<yaml file location> --model_path=<model path>
```

Example:
```bash
python inference.py --training_config=il_config.yaml --model_path=logs/HGDAgger_model.pkl
```
