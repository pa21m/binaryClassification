# Diabetes Classification: Comparing Logistic Regression, K-Nearest Neighbors, and Random Forest

## Project Overview

This project aims to compare the performance of three popular classification algorithms—Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest—on the **Pima Indians Diabetes Database**. The dataset contains medical data for 768 female patients, including features such as glucose levels, blood pressure, BMI, age, etc., with the goal of predicting whether a patient has diabetes or not (binary classification).

The project includes:
- Data preprocessing
- Model training and evaluation
- Performance comparison of the three classifiers

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Exploration](#data-exploration)
- [Models](#models)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/project-name.git

2. Navigate into the project directory:
   cd project-name
  
3. Install the required dependencies:

   pip install -r requirements.txt
   Required Libraries:
   numpy
   pandas
   matplotlib
   seaborn
   scikit-learn
   kaggle (for downloading the dataset)

## Usage
Running the Program
To run the program, execute the following command:

python diabetes_classification.py
The program will load the Pima Indians Diabetes dataset, perform preprocessing, train the three classification models, and output the performance evaluation.

## Data Exploration
Before model training, exploratory data analysis (EDA) is performed, including:

Visualizing feature distributions

Checking for missing or zero values in critical features (e.g., glucose, blood pressure)

Generating correlation matrices to identify relationships between features and the outcome variable.

## Models
1. Logistic Regression
A linear model that predicts the probability of a binary outcome.

Trained using the standard logistic regression technique from scikit-learn.

2. K-Nearest Neighbors (KNN)
A non-parametric method that classifies based on the majority label of the k closest training samples.

The optimal k value is determined using error rates.

3. Random Forest
A powerful ensemble learning method based on decision trees, which improves performance by averaging multiple decision trees trained on random subsets of the data.

## Results
Performance Metrics
Each model is evaluated using the following metrics:

Accuracy: The proportion of correctly classified instances.

Precision: The proportion of true positive predictions relative to all positive predictions.

Recall: The proportion of true positive predictions relative to all actual positives.

F1 Score: The harmonic mean of precision and recall.

Confusion Matrix: Visualizes the classification results.

## Conclusion
After training and evaluating all three classifiers, Random Forest achieves the highest accuracy and overall performance compared to Logistic Regression and K-Nearest Neighbors.

License
This project is licensed under the MIT License 
