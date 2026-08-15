# Titanic Machine Learning Classification Pipeline

## Project Overview

This project implements an end-to-end machine learning classification pipeline in Python using the Titanic dataset.

The objective is to predict whether a passenger survived the Titanic disaster based on passenger information such as age, sex, passenger class, fare, and family relationships.

The project demonstrates data loading, data preprocessing, train/test splitting, model training, model evaluation, and model persistence using a clean modular project structure.

## Machine Learning Model

The classification model used in this project is a **Random Forest Classifier** from scikit-learn.

The model is configured with 100 estimators and a fixed `random_state=42` to support reproducible results.

## Data Preprocessing

The following preprocessing steps are performed:

* Duplicate rows are removed.
* `PassengerId`, `Name`, `Ticket`, and `Cabin` are removed from the model features.
* Missing `Age` values are filled using the median age.
* Missing `Embarked` values are filled using the most frequent value.
* Categorical variables such as `Sex` and `Embarked` are converted to numerical features using one-hot encoding.
* The dataset is separated into features and the `Survived` target variable.
* Data is split into 80% training data and 20% testing data.
* Stratified splitting is used to preserve the target class distribution.
* `random_state=42` is used for reproducibility.

## Model Evaluation

The model is evaluated using the following classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Results

The current model produced the following results on the test dataset:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8156 |
| Precision | 0.7812 |
| Recall    | 0.7246 |
| F1-Score  | 0.7519 |

Confusion Matrix:

```text
[[96 14]
 [19 50]]
```

## Project Structure

```text
python-ml-pipeline-assessment/
│
├── data/
│   └── titanic.csv
│
├── models/
│   └── titanic_model.joblib
│
├── notebooks/
├── screenshots/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

The project requires Python 3 and the following libraries:

* pandas
* numpy
* scikit-learn
* joblib
* matplotlib

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd python-ml-pipeline-assessment
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Make sure the Titanic CSV dataset is available at:

```text
data/titanic.csv
```

Run the complete machine learning pipeline:

```bash
python main.py
```

The program will:

1. Load the Titanic CSV dataset.
2. Clean and preprocess the data.
3. Split the data into training and testing sets.
4. Train a Random Forest classification model.
5. Evaluate the model using classification metrics.
6. Display the confusion matrix.
7. Save the trained model.

## Saved Model

The trained model is saved using `joblib` at:

```text
models/titanic_model.joblib
```

This allows the trained model to be loaded later without retraining it.

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* joblib
* Git / GitHub
