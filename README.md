# NSGA-II + PPO Spectrum Sharing in 6G CRN

This repository contains the source code, pretrained models, datasets, and evaluation scripts for the NSGA-II + PPO hybrid spectrum sharing framework in 6G Cognitive Radio Networks.

## Project Structure

- `code/` - Python source code and evaluation scripts  
- `models/` - Pretrained model weights  
- `data/` - Sample datasets for evaluation  
- `results/` - Generated evaluation results and plots  
- `notebooks/` - Colab notebooks for easy evaluation

## Evaluation Using Google Colab

To evaluate the model easily without setup, open and run the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ancilla-chgs/nsga2-ppo-6g-crn/blob/main/notebooks/evaluate_colab.ipynb)

This notebook will:

- Install dependencies  
- Clone this repo  
- Run the evaluation script on the default validation dataset  
- Save and display evaluation results

---

## How to Run Evaluation Locally

You can also run evaluation locally after cloning the repo:

```bash
pip install -r requirements.txt
python code/evaluate.py --dataset data/val_dataset_sinr.csv
