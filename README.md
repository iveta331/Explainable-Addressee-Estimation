# Explainable Addressee Estimation

## Installation: 
Use conda virtual environment (python=3.8).  
First, install PyTorch according to your version of CUDA. 
Please note that the current version of PyTorch is 
not compatible with python 3.8 therefore use one of the 
older ones. We used 2.0.1 (`conda install pytorch==2.0.1 
torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia`)
Then use pip to install packages from requirements.txt 
(`pip install -r requirements.txt`).  
After that you also need to install wandb 
(`conda install -c conda-forge wandb`) and 
pyrallis (`pip install pyrallis`), which weren't 
included in the previous versions of the code,
therefore are not included in requirements.txt.  
Another required package is timm for the vision transformer (`pip install timm`).
Lastly, the XAE package itself needs to be installed as:  
```
cd path_to_X-AE/X-AE  
python -m pip install -e .
```

## Project structure:
Considering the current structure:
```
└── X-AE
    ├── XAE
    │   ├── BASE
    │   ├── PHASE1
    │   └── PHASE2
    ├── scripts
    │   ├── phase1
    │   └── phase2
    │       ├── ablations
    │       ├── att1
    │       ├── att2
    │       └── att_comb
    ├── models
    ├── data
    │   └── dataset_slots
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── requirements_old.txt
    └── setup.py
```
please note the location of folders `data` and `models` 
(which are included in `.gitignore` and therefore are not 
on GitHub).  
If you want to use different file structure, it is necessary to 
modify parameters `data_dir` and `models_dir` in `utils.py` 
accordingly.  

## Used methods:

In `scripts/phase1` we provide codes used for optimising the IAE network. 
All explainable networks are in `scripts/phase2`. Here, folders
`att1` and `att2` contain additional methods described in appendix.
`att_comb` containes codes for the XAE model. In `ablations` we
define other combinations of explainable/non-explainable modules
used in the ablation study.

## Used data:

We are using Vernissage dataset pre-processed according to Mazzola et al.

```
C. Mazzola, M. Romeo, F. Rea, A. Sciutti and A. Cangelosi, 
"To Whom are You Talking? A Deep Learning Model to Endow Social Robots with Addressee Estimation Skills," 
2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-10, 
doi: 10.1109/IJCNN54540.2023.10191452.
```
