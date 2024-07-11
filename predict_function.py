
def predict(models, X_train, X_test, y_train, y_test):
    for name, model in models:
        model.fit(X_train, y_train)
    
        print(f"{name}")
        y_hat = model.predict(X_train)
    
        print(f"Training error: {mean_absolute_error(y_train, y_hat):.2f}")
        print(f"Training accuracy: {r2_score(y_train, y_hat):.2f}")
    
        print("_"*100)
    
        y_hat = model.predict(X_test)
    
        print(f"Testing error: {mean_absolute_error(y_test, y_hat):.2f}")
        print(f"Testing accuracy: {r2_score(y_test, y_hat):.2f}")
        print()
