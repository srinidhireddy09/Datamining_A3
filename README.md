# Datamining_A3
This repository contains the work for a comprehensive assignment that involves association rule generation and deep learning tasks. The assignment is divided into three main tasks:

Association Rule Generation from Transaction Data
Image Classification using Convolutional Neural Networks (CNN)
Text Classification by Fine-Tuning a Pre-trained BERT Model for Multi-label Classification
 ## Table of contents
Association Rule Generation
Image Classification using CNN
Text Classification by Fine-tuning BERT
Setup and Installation
Files and Directories
Experiment and Results
1. Association Rule Generation from Transaction Data
Task Overview
This task involves generating association rules from a transaction dataset. Specifically, you will:

Analyze a dataset containing grocery items transactions.
Generate association rules using the Mlxtend library.
Perform analysis for different support and confidence thresholds.
Steps Taken:
Data Loading:
Downloaded the transaction dataset from the provided Google Drive link, using the dataset number assigned.

 ## Exploration

Found the number of unique items and transactions in the dataset.
Identified the most popular item and the number of transactions containing that item.
Association Rule Generation:

Used Apriori algorithm to generate frequent itemsets with a minimum support threshold of 0.01 and confidence threshold of 0.08.
Parameter Tuning:

Experimented with various support and confidence values and visualized the results using a heatmap.
Libraries Used:

Mlxtend, pandas, matplotlib, seaborn.
2. Image Classification using CNN
Task Overview
This task involves constructing a 4-class classification model using a Convolutional Neural Network (CNN). The architecture includes:

Two convolutional layers with different filter sizes.
Max-pooling layers.
A fully connected hidden layer and an output layer with 4 nodes (since there are 4 classes).
Steps Taken:
## Data Preprocessing

Loaded and preprocessed the image dataset.
Used data augmentation techniques to improve model performance.
Model Construction:

Implemented a CNN with the following architecture:
Convolutional Layer 1: 8 filters of size 3x3.
Max Pooling 1: Pool size of 2x2.
Convolutional Layer 2: 4 filters of size 3x3.
Max Pooling 2: Pool size of 2x2.
Flatten layer and fully connected layer.
Output layer with 4 nodes (softmax activation).
Experimentation:

Based on the last digit of my Rowan Banner ID, I modified the second convolutional layer to experiment with different filter sizes or numbers of filters.
Training:

Trained the model using the Adam optimizer and categorical cross-entropy loss for 20 epochs.
Plotted the learning curves showing training and validation accuracy over epochs.
Evaluation:

Compared the performance of the original model with the modified models (based on the experiment chosen).
Discussed overfitting, underfitting, or correct fitting of the models.
## Libraries Used:

TensorFlow/Keras, matplotlib, seaborn.
3. Text Classification by Fine-Tuning BERT for Multi-label Classification
Task Overview
In this task, you will fine-tune a pre-trained BERT model (bert-base-uncased) for multi-label text classification on the tweet dataset. The goal is to predict multiple labels for each tweet (e.g., anger, fear, joy).

Steps Taken:
Data Preprocessing:

Loaded and preprocessed the dataset containing tweet texts and corresponding labels.
Tokenized the tweet texts using BertTokenizer.
Model Fine-tuning:

Fine-tuned the BERT base uncased model for multi-label classification with a training set of 3000 records and 5 epochs.
Evaluation:

Calculated the test accuracy using the approach where all labels must match.
Modified the accuracy calculation such that a prediction is correct if at least one label matches.
Learning Curves:

Plotted training and validation loss curves for 5 epochs.
Libraries Used:

Transformers, Torch, matplotlib, seaborn.
Setup and Installation
Prerequisites:
To run the code in this repository, make sure you have the following Python libraries installed:

tensorflow or torch (depending on which framework you prefer)
mlxtend
pandas
matplotlib
seaborn
transformers
You can install them via pip:

bash
Copy code
pip install tensorflow mlxtend pandas matplotlib seaborn transformers torch
Steps:
Clone this repository:
bash
Copy code
git clone https://github.com/your-username/DeepLearning-AssociationAnalysis.git
cd DeepLearning-AssociationAnalysis
Download the necessary dataset files as described in the individual tasks and place them in the appropriate folders.

Run the scripts corresponding to each task:

For association rule generation, run association_rule_generation.py.
For CNN classification, run cnn_classification.py.
For BERT text classification, run bert_text_classification.py.
Files and Directories
The structure of the repository is as follows:

graphql
Copy code
DeepLearning-AssociationAnalysis/
│
├── association_rule_generation.py         # Script for association rule generation
├── cnn_classification.py                  # Script for CNN image classification
├── bert_text_classification.py            # Script for fine-tuning BERT
├── data/                                  # Folder containing dataset files
│   ├── grocery_items_{DATASET_NUMBER}.csv
│   ├── train_images/                      # Folder with training images for CNN task
│   ├── test_images/                       # Folder with test images for CNN task
│   └── tweets.json                        # JSON file containing tweet data for BERT task
├── results/                               # Folder to save output plots
│   ├── learning_curves.png                # Plot for CNN model learning curves
│   └── heatmap.png                        # Heatmap for association rule results
└── README.md                              # This README file
## Experiment and Results
Association Rule Generation Results:
Heatmaps showing the number of association rules extracted for various combinations of support and confidence values.
Visualization of the most popular item in the dataset.
CNN Classification Results:
Learning curves showing training and validation accuracy for different filter sizes or number of filters.
Discussion on model performance and comparison between the baseline and modified models.
Text Classification Results:
Learning curves showing training and validation losses during BERT fine-tuning.
Accuracy results using different definitions of correctness for multi-label classification.
Conclusion
This assignment covers important aspects of both association analysis and deep learning tasks, including rule mining, CNN model training, and fine-tuning pre-trained models like BERT. Through these tasks, we explore various machine learning techniques, parameter tuning, and performance evaluation.
