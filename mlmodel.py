path='/content/drive/MyDrive/Mental Health Dataset.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
df = pd.read_csv(path)


# Display the first few rows of the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Handle missing values (if any)
df['self_employed'].fillna(df['self_employed'].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Set the target variable and feature matrix
y = df['mental_health_interview']
X = df.drop('mental_health_interview', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

pickle.dump(rfc,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))