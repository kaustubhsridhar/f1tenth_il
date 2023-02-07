# F1TENTH Imitation Learning

This repository contains code associated with [A Benchmark Comparison of Imitation Learning-based Control Policies for Autonomous Racing
](https://arxiv.org/abs/2209.15073)

## Docker
Build and run.
```bash
cd f1tenth_il
docker build -t f1tenth_il_docker .
docker run -it --rm -e DISPLAY=$DISPLAY:0 f1tenth_il_docker
```

Note: use pip3 and python3 inside container. Example:
Example:
```bash
python3 train.py --algorithm=hdagger --training_config=il_config.yaml
```

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
conda create -n f110_il python=3.8
```

Activate the environment
```bash
conda activate f110_il
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
pip install -r requirements.txt
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
python train.py --algorithm=hg-dagger --training_config=il_config.yaml
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
