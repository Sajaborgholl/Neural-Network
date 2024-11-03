import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Clean column names and entries
    data.columns = data.columns.str.strip()
    for column in data.columns:
        data[column] = data[column].astype(str).str.strip("'").str.strip()

    # Handle missing values
    data['node-caps'].fillna('unknown', inplace=True)
    data['breast-quad'].fillna(data['breast-quad'].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in data.columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Separate features and target variable
    X = data.drop('class', axis=1)
    y = data['class']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders
