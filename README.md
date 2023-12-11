# Student Performance Prediction Project

## Overview
This project aims to predict student performance based on various features such as job, study time, failures, absences, and first and second period grades. The project utilizes a linear regression model from the `scikit-learn` library in Python.

## Getting Started

### Prerequisites
- Python 3.x
- PyCharm (or any other IDE)

### Installation
1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/pramodyasahan/grade-predictor
   ```

2. **Install Required Libraries**:
   - `pandas` for data manipulation and analysis
   - `numpy` for numerical operations
   - `matplotlib` for plotting graphs
   - `scikit-learn` for machine learning tools

   You can install these using pip:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

3. **Download the Dataset**: 
   Download the dataset from [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) and place it in the project directory.

## Project Structure

- `student-mat.csv`: The dataset file.
- `main.py`: The main Python script with data preprocessing, training, and evaluation.

## Running the Project

1. Open the project in PyCharm.
2. Ensure that the dataset `student-mat.csv` is in the correct directory.
3. Run `main.py` to start the training and evaluation process.

## Code Explanation

### Data Preprocessing
- **Feature Selection**: Selects specific columns from the dataset as features.
- **One-Hot Encoding**: Applies one-hot encoding to categorical features.
- **Data Splitting**: Splits the dataset into training and testing sets.

### Model Training
- **Linear Regression**: Uses the `LinearRegression` class from `scikit-learn` to train the model on the training set.

### Prediction and Evaluation
- **Predicting Grades**: The model predicts grades on the test set.
- **Comparison Plot**: A plot is generated to compare predicted grades against actual grades.

## Theoretical Background

- **Linear Regression**: A statistical method that models the relationship between a dependent variable and one or more independent variables.
- **One-Hot Encoding**: A process of converting categorical data variables so they can be provided to machine learning algorithms to improve predictions.
- **Training and Testing Split**: This concept involves dividing the dataset into two parts: training data to train the model, and testing data to evaluate its performance.

## Contribution
Feel free to fork this repository and contribute to its development. Any contributions you make are greatly appreciated.
