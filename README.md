
# Breast Cancer Classification with Ensemble Learning

This repository contains a machine learning project to classify breast cancer diagnoses as either benign or malignant using an ensemble of classifiers: Random Forest, Gradient Boosting, and Support Vector Machine (SVM). The classifiers are combined using a VotingClassifier to enhance the model's performance and generalize better on unseen data.

## Project Overview

The project uses the Breast Cancer dataset to train and evaluate a machine learning model that predicts the malignancy of breast cancer based on various features.

### Key Features:
- **Ensemble Learning**: Combines multiple models (Random Forest, Gradient Boosting, and SVM) for improved performance.
- **Data Preprocessing**: Handles missing values, scales features, and encodes the target variable.
- **Model Evaluation**: Evaluates the model using accuracy, precision, recall, F1 score, and confusion matrix.
- **Prediction for New Data**: Allows for making predictions on new input data.
- **Visualization**: Includes visualizations of feature importance and model performance (confusion matrix).

## Requirements

The following libraries are required to run this project:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

The project uses the Breast Cancer dataset (CSV format) for classification tasks. The dataset contains various features about breast cancer tumors, and the target variable is `diagnosis`, indicating whether the tumor is malignant (`M`) or benign (`B`).

## How to Use

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Replace the dataset path:
Make sure to replace the dataset path in the code (`/content/BreastCancer.csv`) with the actual path to your dataset.

### 3. Run the script:
Run the Python script to load the dataset, preprocess it, train the model, and evaluate its performance:

```bash
python breast_cancer_classification.py
```

### 4. Results:
- Predictions for the test set are saved to `breast_cancer_predictions.csv`.
- The VotingClassifier structure is saved as an HTML file (`voting_classifier.html`).
- The confusion matrix and feature importance plots will be displayed.

## File Structure

- `breast_cancer_classification.py`: Main Python script that runs the machine learning model.
- `breast_cancer_predictions.csv`: CSV file containing the model's predictions.
- `voting_classifier.html`: HTML file that visualizes the VotingClassifier structure.
- `README.md`: This file.

## Model Evaluation

The model's performance is evaluated using the following metrics:
- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of true positives among the predicted positives.
- **Recall**: The proportion of true positives among the actual positives.
- **F1 Score**: The harmonic mean of precision and recall.

Additionally, the confusion matrix is displayed to visualize the number of true positives, true negatives, false positives, and false negatives.

## Example Output

| Actual Diagnosis | Predicted Diagnosis |
|------------------|---------------------|
| Benign           | Benign              |
| Malignant        | Malignant           |
| Malignant        | Malignant           |
| Benign           | Benign              |

