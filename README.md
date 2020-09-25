# attack-graph-code

### Installation
You need **Python3.7** and conda to install required packages. 
1. First, clone the project folder to your computer.
2. Then, create a conda environment and activate it:
  ```
  conda create -n attack-graph python=3.7
  conda activate attack-graph
  ```
3. After the environment is activated, install the following required packages:
   ```
   conda install numpy scipy networkx pandas matplotlib ipython jupyter
   conda install pytorch torchvision -c pytorch
   ```
   
4. Install the package to simulate SIS/SIR dynamics
  ```
  pip install EoN
  ```
  

### Reference
This code is used to reproduce the experiments in the following paper:
```
@article{yu2020optimizing,
  title={Optimizing Graph Structure for Targeted Diffusion},
  author={Yu, Sixie and Torres, Leonardo and Alfeld, Scott and Eliassi-Rad, Tina and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:2008.05589},
  year={2020}
}
```
