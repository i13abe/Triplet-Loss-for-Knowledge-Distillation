# Triplet Loss for Knowledge Distillation
An implementation of "**Triplet Loss for Knowledge Distillation**" in Pytorch.

The paper is here https://arxiv.org/abs/2004.08116

## How to use
### 1. clone the all files to a directory
### 2. Intall requirements for python
```sh
pip install requirements
```

And you install pytorch from official page https://pytorch.org/

### 3. Train the teacher model
Modify the settings for training until 4th cell on **teacher.ipynb** and then run it on the jupyter (If you don't use jupyter, please run the **teacher.py**)


### 4. Train the student model
Modify the settings for training until 4th cell on **student.ipynb** and then run it on the jupyter (If you don't use jupyter, please run the **student.py**)

## Execution environment
- OS : Ubuntu 20.04 LTS
- Python : v 3.7.3
- pytorch 1.12.0
