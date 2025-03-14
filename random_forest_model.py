from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))