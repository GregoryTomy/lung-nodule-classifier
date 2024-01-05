# Automated Lung Cancer Detection Through Deep Learning

## Motivation
Lung cancer remains the predominant cause of cancer-related mortality on a global scale. Despite technological advances, early detection of lung cancer remains a challenge, often requiring highly specialized manual review of Computed Tomography (CT) scans. This project aims to develop an automated method for identifying potential malignancies in lung CT scans. Working on this problem offers the opportunity to delve deep into state-of-the-art deep learning frameworks, particularly PyTorch.

## Project Scope
This project proposes an end-to-end deep learning solution (CNNs) for identifying and classifying lung nodules as benign or malignant, based on the [LUNA16 Grand Challenge](https://luna16.grand-challenge.org) dataset. It will have 3 stages:
1. Segmentation: Deploy a model to identify potential nodule locations in CT scans.
2. Nodule Candidate Classification: Implement a model to classify regions of interest as nodule or non-nodule.
3. Malignancy Classification: Implement a model to classify identified nodules as benign or malignant.


## Part 1: Segmentation of CT Scans (Pending).
## Part 3: Classification of Nodules. (Pending)

## Part 2: Classification of Candidate Nodules.
The paper detailing the project methodologies, results, and discussions can be found here: [CSCI_5502_Project.pdf](CSCI_5502_Project.pdf).

[[Watch the presentation]](https://youtu.be/ovUrZZ6Hyj8)


## Results

### Results Table

| Metric    | Value    |
|-----------|----------|
| FROC      | 99.27%   |
| F2        | 12.46%   |
| Recall    | 95.51%   |
| FPR       | 6.91%    |
| Precision | 2.78%    |


### Confusion Matrix

|                  | Predicted Positive | Predicted Negative |
|------------------|--------------------|--------------------|
| **Actual Positive** | 149                | 7                  |
| **Actual Negative** | 5207               | 70135              |


### FROC Curve
![FROC Curve](images/froc.jpg)
*The FROC curve illustrates the trade-off between sensitivity and the average number of false positives per scan, showcasing the performance of our model.*
