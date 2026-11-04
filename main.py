from src.credit_risk_assessment_ml_model.data_ingestion import load_data
from src.credit_risk_assessment_ml_model.data_preprocessing import preprocess
from src.credit_risk_assessment_ml_model.model_build import model

def main():
    # Load the data
    data = load_data()
    print("Data loaded successfully.")

    # Preprocess the data
    X_train,X_test,y_train,y_test = preprocess(data)

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    report = model(X_train,X_test,y_train,y_test)
    print(report)
    
    
main()
