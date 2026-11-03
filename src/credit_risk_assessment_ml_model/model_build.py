from sklearn.ensemble import RandomForestClassifier


def model(X_train,X_test,y_train,y_test):
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    