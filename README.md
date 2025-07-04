# EXPHY

## Visualization Demos
### Explain observation
<!Observation| Reconstruction | Counterfactual 1 | Counterfactual 2
:--------------------------------------------------:|:--------------------------------------------------: |:--------------------------------------------------: |:--------------------------------------------------: 
![image](results/collision/observation.gif)  |  ![image](results/collision/explain.gif) | ![image](results/collision/counterfactual_1.gif) | ![image](results/collision/counterfactual_2.gif) 
|| |
![image](results/charge/observation.gif)  |  ![image](results/charge/explain.gif) | ![image](results/charge/counterfactual_1.gif) | ![image](results/charge/counterfactual_2.gif)>
![image](demo/col_explain.gif)
### Explain counterfactual imagination
![image](demo/col_counterfactual.gif)

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
The commands output force analyses which encompasses three specific parameters: Velocity, Collision Acceleration, and Coulomb Acceleration:
```
python explain.py --scenario collision
python explain.py --scenario charge
```

