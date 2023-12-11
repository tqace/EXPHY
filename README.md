# EXPHY

## Introduction
This is the official PyTorch implementation for the paper: **EXPHY: Learning Explainable Intuitive Physics in Neural Networks**. This repository contains:
- pre-trained model
- Visual demos of **counterfactual imagination**:
  1. Collision: _1. What if the cylinder is heavier?_; _2. What if the sphere is heavier?_
  2. Coulomb interaction: _1. What if the objects are uncharged?_; _2. What if the objects undergo repulsion?_
- Inference code for physical event **explanation**

## Visualization Demos
Observation| Reconstruction | Counterfactual 1 | Counterfactual 2
:--------------------------------------------------:|:--------------------------------------------------: |:--------------------------------------------------: |:--------------------------------------------------: 
![image](results/collision/observation.gif)  |  ![image](results/collision/explain.gif) | ![image](results/collision/counterfactual_1.gif) | ![image](results/collision/counterfactual_2.gif) 
|| |
![image](results/charge/observation.gif)  |  ![image](results/charge/explain.gif) | ![image](results/charge/counterfactual_1.gif) | ![image](results/charge/counterfactual_2.gif) 

## Requirements
- Python 3.6.9
- CUDA 11.0
- Others (See requirements.txt)
  
## Installation 
```
conda create -n exphy python=3.6.9
conda activate exphy
git clone https://github.com/tqace/EXPHY.git && cd EXPHY
pip -r requirments.txt
```
## Explanation inference
The commands output force analyses corresponding to the observations and counterfactual imaginations in the "Visualization Demos".
```
python explain.py --scenario collision
python explain.py --scenario charge
```
The force analysis initiates from the second frame of the input video. The analysis encompasses three specific parameters: Velocity, Collision Acceleration, and Coulomb Acceleration. The resulting table provided by EXPHY will be structured as follows:

| **Frame** | **Velocity** | **Collision Acceleration** | **Coulomb Acceleration** |
|:---------:|:------------:|:----------------------------------:|:---------------------------------:|
| **n-1**   | ... | ... | ... |
| **n**     | object1, object2 | object1, object2 | object1, object2 |
| **n+1**   | ... | ... | ... |

