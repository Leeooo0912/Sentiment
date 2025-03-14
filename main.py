import data
from logistic_regression_model import train_logistic_regression
from random_forest_model import train_random_forest
# Import other model files as you create them

def main():
    # Get preprocessed data
    X_train = data.X_train
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test
    
    # Execute different models
    print("\nTraining Logistic Regression Model...")
    train_logistic_regression(X_train, X_test, y_train, y_test)
    
    print("\nTraining Random Forest Model...")
    train_random_forest(X_train, X_test, y_train, y_test)
    
    # Add more model executions as you create them
    # print("\nTraining SVM Model...")
    # train_svm(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()