````markdown
# FullMedSeg

This repository provides the implementation of **MedSeg**.

---

## Experimental Setup

All experiments are implemented based on the official **U-Bench** open-source codebase to ensure consistent training pipelines and fair comparisons.

Following the U-Bench protocol, all models are trained using **stochastic gradient descent (SGD)** with a momentum of **0.9** and a weight decay of **1×10⁻⁴**.  
The initial learning rate is set to **0.01**, and all models are trained for **300 epochs** with a **batch size of 8**.  
A fixed random seed of **41** is used for all experiments.

### Loss Function

For all **2D datasets**, we adopt a combination of **binary cross-entropy (BCE)** loss and **Dice loss**, defined as:

```math
\mathcal{L} = 0.5 \times \mathrm{BCE}(\hat{y}, y) + \mathrm{Dice}(\hat{y}, y)
````
For **3D datasets**, including **ACDC** and **Synapse**, we follow the U-Bench configuration and apply a weighted Dice formulation:

```math
\mathcal{L} = 0.5 \times \mathrm{BCE}(\hat{y}, y) + 0.7 \times \mathrm{Dice}(\hat{y}, y)
```

---

## Quick Start

### Environment Setup

```bash
source ~/.bashrc
source activate ubench
```

### Training (In-domain: BUSI)

```bash
python main.py \
  --max_epochs 300 \
  --gpu 0 \
  --batch_size 8 \
  --model FullMedSeg \
  --base_dir ./data/busi \
  --dataset_name busi
```

```
```
