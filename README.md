# Loan Eligibility Prediction

This project aims to predict loan eligibility using machine learning techniques. The dataset used for this project was sourced from Kaggle and contains information about loan applicants, including demographics, financial history, and loan status.

## Dataset

The dataset was imported from Kaggle and consists of several features such as:

- Gender

- Marital status

- Education

- Applicant's income

- Co-applicant's income

- Loan amount

- Loan term

- Credit history

- Loan status (target variable)

## Programming Language and Libraries

The project is implemented in Python using popular machine learning libraries such as pandas, scikit-learn, and matplotlib. These libraries are widely used for data manipulation, model training, and visualization in machine learning projects.

## Methodology

The project follows these basic steps:

Data Import: The dataset is imported from Kaggle using pandas' read\_csv() function.
Data Preprocessing: Data preprocessing steps include handling missing values, encoding categorical variables, and feature scaling.
Model Training: A logistic regression model is trained using scikit-learn's LogisticRegression class.
Model Evaluation: The model is evaluated using accuracy, precision, recall, and F1-score metrics.
Prediction: Loan eligibility is predicted using the trained logistic regression model.

## Model Evaluation Results

- **Accuracy**: 1.0

- **Classification Report**:

```
                    precision    recall  f1-score   support

        COLLECTION       1.00      1.00      1.00        24
COLLECTION_PAIDOFF       1.00      1.00      1.00        24
           PAIDOFF       1.00      1.00      1.00        52

          accuracy                           1.00       100
         macro avg       1.00      1.00      1.00       100
      weighted avg       1.00      1.00      1.00       100
```
- **Confusion Matrix**:

```
[[24  0  0]
 [ 0 24  0]
 [ 0  0 52]]
```

## Download report

[View first then download](https://github.com/niladrridas/ml-loan-eligibility/blob/main/MyMLReport_LPU.pdf) 

## Usage

To run the project:

1. Clone the repository to your local machine.

2. Install the required libraries using \`pip install -r requirements.txt\`.

3. Run the main script \`loan\_eligibility\_prediction.py\`.

4. View the results and predictions.

## References

- Dataset: [Kaggle - Loan Eligibility Dataset](https://www.kaggle.com/datasets/zhijinzhai/loandata)

- scikit-learn documentation: [scikit-learn](https://scikit-learn.org/stable/)

- pandas documentation: [pandas](https://pandas.pydata.org/docs/)
