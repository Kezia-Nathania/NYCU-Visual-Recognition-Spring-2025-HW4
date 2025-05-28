# NYCU-Visual-Recognition-2025-Spring-HW4
StudentID  : 312540021  
Name       : Kezia Nathania (林紀霞)  

## Introduction 
The task of this homework is image restoration for two types of degradation, rain and snow. The dataset is split into training/validation with 1600 pairs of clean and degraded images for each degradation type, and the testing set consists of 50 degraded images for each type. This restoration task is evaluated using PSNR (Peak Signal-to-Noise Ratio).  

## How to Install
### Clone the repository:  
  git clone https://github.com/Kezia-Nathania/NYCU-Visual-Recognition-2025-Spring-HW4.git  
  cd NYCU-Visual-Recognition-2025-Spring-HW4  
### Install dependencies:  
  pip install -r requirement.txt  
### Run the train.py file  
  python train.py --epochs 55 --ckpt_dir "checkpoint_dir_path"
### Run the testing.py file  

## Performance Snapshot
![Alt text](PerformanceSnapshot.png)
