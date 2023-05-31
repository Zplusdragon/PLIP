# PLIP
**PLIP** is a novel **L**anguage-**I**mage **P**re-training framework for generic **P**erson representation learning which benefits a range of downstream multimodal person-centric tasks. To explicitly build fine-grained cross-modal associations and mine latent person-centric semantics, we specifically design three pretext tasks: 1) Semantic-fused image colorization, 2) Visual-fused Attribute Prediction and 3) Vision-language Matching.

Also, we present a large-scale person dataset named **SYNTH-PEDES**, where the Stylish Pedestrian Attributes-union Captioning method **(SPAC)** is proposed to synthesize diverse textual descriptions. 

Experiments show that our model not only significantly improves existing methods on downstream tasks, but also shows great ability in the few-shot and domain generalization settings. More details can be found at our paper [PLIP: Language-Image Pre-training for Person Representation Learning](http://export.arxiv.org/abs/2305.08386).

<div align="center"><img src="assets/abstract.png" width="600"></div>

## News
* 🔥[05.31] The pre-trained model checkpoints and zero-shot inference code have been released !
* 🔥[05.31] We released the SYNTH-PEDES dataset !

## SYNTH-PEDES 
SYNTH-PEDES is by far the largest person dataset with textual descriptions without any human annotation effort. Every person image has 2 or 3 different descriptions. The dataset will be released soon.

These are some examples of our SYNTH-PEDES dataset.
<div align="center"><img src="assets/examples.png" width="900"></div>

This is the comparison of SYNTH-PEDES with other popular datasets.
<div align="center"><img src="assets/SYNTH-PEDES.png" width="900"></div>

## Models
We utilize ResNet50 and Bert as our encoders. After pre-training, we fine-tune and evaluate the performance on three downstream tasks. The checkpoints have been released at [Baidu Yun](https://pan.baidu.com/s/1LjT-x6kjGwpO2EP4Ni7bCA?pwd=1037).

### Evaluate on CUHK-PEDES dataset (Text Re-ID R@1/R@10)
| Pre-train | CMPM/C | SSAN | LGUR | 
| :---: |:---: |:---: | :---: 
| IN sup | 54.81/83.22 |  61.37/86.73 | 64.21/87.93 
| IN unsup |55.34/83.76| 61.97/86.63| 65.33/88.47
| CLIP |55.67/83.82| 62.09/86.89| 64.70/88.76
| LUP |57.21/84.68| 63.91/88.36| 65.42/89.36
| LUP-NL |57.35/84.77| 63.71/87.46| 64.68/88.69
| **PLIP(ours)** |**69.23/91.16**| **64.91/88.39**| **67.22/89.49**

### Evaluate on ICFG-PEDES dataset (Text Re-ID R@1/R@10)
| Pre-train | CMPM/C | SSAN | LGUR | 
| :---: |:---: |:---: | :---: 
| IN sup | 47.61/75.48| 54.23/79.53| 57.42/81.45
| IN unsup |48.34/75.66| 55.27/79.64| 59.90/82.94
| CLIP |48.12/75.51| 53.58/78.96| 58.35/82.02
| LUP |50.12/76.23| 56.51/80.41| 60.33/83.06
| LUP-NL |49.64/76.15| 55.59/79.78| 60.25/82.84
| **PLIP(ours)** |**64.25/86.32**| **60.12/82.84**| **62.27/83.96**

### Evaluate on Market1501 & DukeMTMC (Image Re-ID mAP/cmc1)
| Methods | Market1501 | DukeMTMC | 
| :---: |:---: |:---: 
| BOT | 85.9/94.5 |76.4/86.4
| BDB |86.7/95.3| 76.0/89.0
| MGN |87.5/95.1 |79.4/89.0
| ABDNet |88.3/95.6| 78.6/89.0
| **PLIP+BOT** | 88.0/95.1| 77.0/86.5
| **PLIP+BDB** |88.4/95.7| 78.2/89.8
| **PLIP+MGN** |90.6/96.3| **81.7**/90.3
| **PLIP+ABDNet**|**91.2**/**96.7** |81.6/**90.9**

### Evaluate on PETA & PA-100K & RAP (PAR mA/F1)
| Methods | PETA | PA-100K | RAP
| :---: |:---: |:---: |:---: 
| DeepMAR | 80.14/83.56| 78.28/84.32| 76.81/78.94
| Rethink |83.96/86.35 |80.21/87.40 |79.27/79.95
| VTB |84.12/86.63| 81.02/87.31| 81.43/80.63
| Label2Label |84.08/86.57 |82.24/87.08| 81.82/80.93
| **PLIP+DeepMAR** | 82.46/85.87 |80.33/87.24 |78.96/80.12
| **PLIP+Rethink**|85.56/87.63| 82.09/88.12| 81.87/81.53
| **PLIP+VTB** |86.03/**88.14**| 83.24/88.57 |83.64/**81.78**
| **PLIP+Label2Label** |**86.12**/88.08 |**84.36**/**88.63**| **83.77**/81.49


## Usage
### Install Requirements
we use 4 RTX3090 24G GPU for training and evaluation.

Create conda environment
'''
conda create --name PLIP --file requirements.txt
conda activare PLIP
'''

## Reference
If you use PLIP in your research, please cite it by the following BibTeX entry:
```
@misc{zuo2023plip,
    title={PLIP: Language-Image Pre-training for Person Representation Learning},
    author={Jialong Zuo and Changqian Yu and Nong Sang and Changxin Gao},
    year={2023},
    eprint={2305.08386},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
