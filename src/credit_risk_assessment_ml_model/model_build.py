from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import os
import logging  


def model(X_train,X_test,y_train,y_test):
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logging.info("Model training completed successfully.")
    
    y_pred = model.predict(X_test)
    logging.info("Model prediction completed successfully.")
    
    report = classification_report(y_test, y_pred)
    logging.info("Classification report generated successfully.")
    
    os.makedirs('models',exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    logging.info("Model saved successfully.")

    
    return report
    