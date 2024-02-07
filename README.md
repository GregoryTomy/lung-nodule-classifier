# Automated Lung Cancer Detection Through Deep Learning

## Motivation
Lung cancer remains the predominant cause of cancer-related mortality on a global scale. Despite technological advances, early detection of lung cancer remains a challenge, often requiring highly specialized manual review of Computed Tomography (CT) scans. This project aims to develop an automated method for identifying potential malignancies in lung CT scans. Working on this problem offers the opportunity to delve deep into state-of-the-art deep learning frameworks, particularly PyTorch.

## Project Scope
This project proposes an end-to-end deep learning solution (CNNs) for identifying and classifying lung nodules as benign or malignant, based on the [LUNA16 Grand Challenge](https://luna16.grand-challenge.org) dataset. It will have 3 stages:
1. Segmentation: Deploy a model to identify potential nodule locations in CT scans.
2. Nodule Candidate Classification: Implement a model to classify regions of interest as nodule or non-nodule.
3. Malignancy Classification: Implement a model to classify identified nodules as benign or malignant.

The project started out as solely a classification task (stage 2). The paper detailing the project methodologies, results, and discussions can be found here: [CSCI_5502_Project.pdf](CSCI_5502_Project.pdf).

[[Watch the presentation]](https://youtu.be/ovUrZZ6Hyj8)
[](images/thumbnail.jpg)

## Part 1: Segmentation of CT Scans.

### Segmentation Results

| Metric           | Value      |
|------------------|------------|
| Loss             | 0.9377     |
| Precision        | 0.0225     |
| Recall           | 0.8319     |
| F1 Score         | 0.0438     |
| True Positive    | 83.2%      |
| False Negative   | 16.8%      |
| False Positive   | 3618.1%    |

The elevated false positive rate observed in the validation phase is anticipated due to the significant disparity in the dimensions of the datasets used for training and validation. Specifically, the validation dataset encompasses an area of $2^{18}$ pixels, which is notably larger than the $2^{12}$ pixels area utilized for the training dataset. This results in a validation area that is 64 times greater in size compared to the training area. Given this context, our focus was primarily on optimizing the recall metric, with the understanding that any precision-related concerns will be addressed by the downstream classification model. 

### Examples of segmentation output. 

Positive predictions in Green. False positives in Red.

![segmentation_ct_result_1](images/seg_val.png)

![segmentation_ct_result_2](images/seg_val_2.png)

## Part 2: Classification of Candidate Nodules.
[CSCI_5502_Project.pdf](CSCI_5502_Project.pdf).

### Classification Model Architecture
![Classification Model Architecture](images/model_arch.jpeg)

### Results

#### Results Table

| Metric    | Value    |
|-----------|----------|
| FROC      | 99.27%   |
| F2        | 12.46%   |
| Recall    | 95.51%   |
| FPR       | 6.91%    |
| Precision | 2.78%    |


#### Confusion Matrix

|                  | Predicted Positive | Predicted Negative |
|------------------|--------------------|--------------------|
| **Actual Positive** | 149                | 7                  |
| **Actual Negative** | 5207               | 70135              |


#### FROC Curve
![FROC Curve](images/froc.jpg)
*The FROC curve illustrates the trade-off between sensitivity and the average number of false positives per scan, showcasing the performance of our model.*

## Part 3: Classification of Nodules.

### Baseline ROC
To define a performance benchmark, a simple predictive model was constructed to infer the likelihood of malignancy from the size of nodules, premised on the assumption that larger nodules have a higher probability of being malignant. The Receiver Operating Characteristic (ROC) curve was used to find the most effective diameter threshold for this malignancy prediction. The baseline model achieved an Area Under the ROC Curve (AUC) of 0.901.

![Baseline ROC](images/roc_diameter_baseline.png)
*Figure: ROC Curve illustrating the performance of the nodule size predictor. The red values at 5.22 mm and 10.55 mm represent the sensitivity and specificity trade-off at these specific diameter thresholds.*

In Part 3, we refined our Stage 2 lung nodule classification model to predict malignancy. We fine-tuned the final linear layer and last convolutional block, while keeping the earlier weights, to better differentiate between benign and malignant nodules.