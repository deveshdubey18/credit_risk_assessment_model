from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def model(X_train,X_test,y_train,y_test):
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    
    return report
    