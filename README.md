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


## Results
Below are the visual representations of the outcomes from our models:

### Results Table
![Results Table](images/results_table.png)
*The table above outlines the performance metrics of our classification model, including accuracy, precision, recall, and F1 score.*

### Confusion Matrix
![Confusion Matrix](./images/confusion_matrix.png)
*The confusion matrix provides insight into the true positive, false positive, true negative, and false negative rates of the nodule candidate classification model.*

### FROC Curve
![FROC Curve](images/froc.jpg)
*The FROC curve illustrates the trade-off between sensitivity and the average number of false positives per scan, showcasing the performance of our malignancy classification model.*
