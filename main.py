import pandas as pd
from sklearn.model_selection import train_test_split
from models.logistic_regression import SentimentLogisticRegression

def load_and_split_data(data_path='data.csv', test_size=0.2):
    """Load and split data into train and test sets"""
    df = pd.read_csv(data_path)
    X = df['text']
    y = df['score']
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a specified model"""
    if model_name.lower() == 'logistic':
        model = SentimentLogisticRegression()
    else:
        raise ValueError(f"Model {model_name} not implemented")
    
    # Train
    print(f"\nTraining {model_name} model...")
    model.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluation Results:")
    report = model.evaluate(X_test, y_test)
    print(report)
    
    # Save model
    model.save_model(f'models/{model_name.lower()}_model.joblib')
    return model

def test_predictions(model):
    """Test model with some example texts"""
    test_texts = [
        "This is absolutely amazing!",
        "I really hate this product.",
        "The service was okay, nothing special.",
    ]
    
    print("\nTesting predictions:")
    for text in test_texts:
        sentiment, score = model.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment} (score: {score})")

if __name__ == "__main__":
    # Load and split data
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Train and evaluate models
    models_to_train = ['logistic']  # Add more models here as they're implemented
    
    for model_name in models_to_train:
        model = train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test)
        test_predictions(model) 