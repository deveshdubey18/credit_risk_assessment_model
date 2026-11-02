from src.credit_risk_assessment_ml_model.data_ingestion import load_data
from src.credit_risk_assessment_ml_model.data_preprocessing import preprocess


def main():
    # Load the data
    data = load_data()

    # Preprocess the data
    X_train,X_test,y_train,y_test = preprocess(data)

    print(X_train.shape, X_test.shape)
    
    
main()
