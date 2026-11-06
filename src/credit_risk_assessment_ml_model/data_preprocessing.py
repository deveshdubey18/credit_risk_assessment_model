from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from imblearn.over_sampling import SMOTE
import logging

def preprocess(df):
    # Split the data into features and target variable
    X = df.drop('is_at_risk', axis=1)
    y = df['is_at_risk']
    logging.info("Data split into features and target variable.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logging.info("Data split into training and testing sets.")

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Features scaled using StandardScaler.")

    # Handle class imbalance using SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train) # type: ignore
    logging.info("Class imbalance handled using SMOTE.")
    
    logging.info("Data preprocessing completed successfully.")

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test