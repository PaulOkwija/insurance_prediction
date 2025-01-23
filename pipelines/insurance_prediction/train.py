from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error


def evaluate_model(y_train, y_pred, category="Train"):
    """
    Evaluate the model's performance on the given data.

    Parameters:
    y_train (array-like): True labels.
    y_pred (array-like): Predicted labels.
    category (str): Category of the evaluation (default is "Train").

    Prints:
    Accuracy, Balanced Accuracy, and Mean Squared Error of the model.
    """
    train_acc = accuracy_score(y_train, y_pred) 
    train_bal_acc = balanced_accuracy_score(y_train, y_pred)
    train_error = mean_squared_error(y_train, y_pred)

    print(f"{category} Accuracy: {train_acc:.4f}")
    print(f"{category} Balanced Accuracy: {train_bal_acc:.4f}")
    print(f"{category} Mean Squared Error: {train_error:.4f}")