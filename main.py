# Importing necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_dataset(file_path):
    """
    Load the dataset from the specified file path.
    
    Parameters:
        file_path (str): The path to the dataset file.
    
    Returns:
        DataFrame: The loaded dataset.
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the dataset by dropping unnecessary columns and converting date columns.
    
    Parameters:
        data (DataFrame): The dataset.
    
    Returns:
        DataFrame: Features (X).
        Series: Target variable (y).
    """
    try:
        # Convert date columns to datetime format
        data['effective_date'] = pd.to_datetime(data['effective_date'])
        data['due_date'] = pd.to_datetime(data['due_date'])
        data['paid_off_time'] = pd.to_datetime(data['paid_off_time'])

        # Extract useful information from date columns
        data['effective_year'] = data['effective_date'].dt.year
        data['effective_month'] = data['effective_date'].dt.month
        data['effective_day'] = data['effective_date'].dt.day

        data['due_year'] = data['due_date'].dt.year
        data['due_month'] = data['due_date'].dt.month
        data['due_day'] = data['due_date'].dt.day

        data['paid_off_year'] = data['paid_off_time'].dt.year
        data['paid_off_month'] = data['paid_off_time'].dt.month
        data['paid_off_day'] = data['paid_off_time'].dt.day
        
	# Perform one-hot encoding on 'Gender' and 'education' columns
        data = pd.get_dummies(data, columns=['Gender', 'education'], drop_first=True)
        
	# Drop original date columns
        data.drop(['effective_date', 'due_date', 'paid_off_time'], axis=1, inplace=True)
        
        # Drop unnecessary columns
        X = data.drop(['Loan_ID', 'loan_status'], axis=1)  # Drop 'Loan_ID' and 'loan_status' columns
        y = data['loan_status']  # Target variable
        
  	# Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)

        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None

def evaluate_model(y_test, y_pred):
    """
    Evaluate the performance of the model using accuracy, classification report, and confusion matrix.
    
    Parameters:
        y_test (Series): True labels.
        y_pred (array-like): Predicted labels.
    """
    try:
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Generate confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    except Exception as e:
        print(f"Error evaluating model: {e}")

def main():
    # File path to the dataset
    file_path = 'data.csv'  # Replace 'data.csv' with your dataset file path

    # Load the dataset
    data = load_dataset(file_path)
    if data is None:
        return
    
    # Preprocess the data
    X, y = preprocess_data(data)
    if X is None or y is None:
        return
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Creating and training the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predicting loan eligibility
    y_pred = model.predict(X_test)
    
    # Evaluating the model
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
