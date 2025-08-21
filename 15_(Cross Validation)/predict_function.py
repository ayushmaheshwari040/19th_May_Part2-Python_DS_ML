from sklearn.metrics import r2_score, mean_absolute_error

def predict(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a list of models.
    
    Parameters:
    models (list): List of tuples (model_name, model_object)
    X_train, X_test, y_train, y_test: Train/test data
    """
    results = []
    for name, model in models:
        print(f"\nTraining model: {name}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Save results
        results.append((name, r2, mae))
        
        # Print results
        print(f"{name} - R² Score: {r2:.4f}")
        print(f"{name} - MAE: {mae:.2f}")
    
    # Summary table
    print("\nSummary:")
    for name, r2, mae in results:
        print(f"{name:20} | R²: {r2:.4f} | MAE: {mae:.2f}")
