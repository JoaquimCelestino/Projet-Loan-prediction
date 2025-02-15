# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Define file paths
train_path = r"c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\Projet Loan prediction\train_ctrUa4K.csv"
test_path = r"c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\Projet Loan prediction\test_lAUu6dG.csv"

# Load datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Preprocessing function
def preprocess_data(data):
    # Fill missing values
    data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
    data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
    data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
    data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
    data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
    data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        data[col] = le.fit_transform(data[col])
    
    return data

# Preprocess data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Encode target variable
train_data['Loan_Status'] = train_data['Loan_Status'].map({'Y': 1, 'N': 0})

# Define input features and target variable
X = train_data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train_data['Loan_Status']

# Prepare test features
test_features = test_data.drop(['Loan_ID'], axis=1)

# Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(
    X_train, 
    y_train, 
    eval_set=[(X_valid, y_valid)], 
    eval_metric="logloss",
    verbose=False
)

# Validate model
y_pred = xgb_model.predict(X_valid)

# Evaluate model
accuracy = accuracy_score(y_valid, y_pred)
print("Validation Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

# Predict test data
test_predictions = xgb_model.predict(test_features)

# Prepare submission file
submission = test_data[['Loan_ID']].copy()
submission['Loan_Status'] = test_predictions
submission['Loan_Status'] = submission['Loan_Status'].map({1: 'Y', 0: 'N'})
submission.to_csv('Loan_Eligibility_Prediction.csv', index=False)

print("Submission file created: 'Loan_Eligibility_Prediction.csv'")
