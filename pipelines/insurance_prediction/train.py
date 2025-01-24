from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier



def evaluate_model(y_true, y_pred, category="Train"):
    """
    Evaluate the model's performance on the given data.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    category (str): Category of the evaluation (default is "Train").

    Prints:
    Accuracy, Balanced Accuracy, and Mean Squared Error of the model.
    """
    acc = accuracy_score(y_true, y_pred) 
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    error = mean_squared_error(y_true, y_pred)

    print(f"{category} Accuracy: {acc:.4f}")
    print(f"{category} Balanced Accuracy: {bal_acc:.4f}")
    print(f"{category} Mean Squared Error: {error:.4f}")


def train_model(X_data, y_data):
    """
    Trains a Random Forest classifier on the given data.

    Parameters:
    X_data (array-like): Training features.
    y_data (array-like): Training labels.

    Returns:
    RandomForestClassifier: Trained model.
    """
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=100, class_weight='balanced', max_depth=8)
    rf.fit(X_data, y_data)
    print("Train accuracy: ", rf.score(X_data, y_data))
    return rf