from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import pandas as pd


def feature_select(x_train, y_train, feature_names, top_features=5):
    """
    Selects important features from the training data using a Random Forest classifier.
    
    Parameters:
    x_train (pd.DataFrame or np.ndarray): The training input samples.
    y_train (pd.Series or np.ndarray): The target values.
    train_features (pd.DataFrame): The training features.
    top_features (int): The number of top features to select.
    
    Returns:
    np.ndarray: An array indicating which features are selected.
    """

    # Using random forest to fit data
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
    rf.fit(x_train, y_train)
    print("Rf score", rf.score(x_train, y_train))

    # Extract feature importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Rank features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    print(feature_importance_df.to_string())

    # Select top N top_features (example selecting top 5 features)
    top_features = feature_importance_df['Feature'][:top_features].values

    return top_features

def split_data(features, labels, val_size, strat=None, seed=42):
    """
    Splits the data into training and validation sets.
    
    Parameters:
    train_features (pd.DataFrame or np.ndarray): The input features.
    train_y (pd.Series or np.ndarray): The target values.
    val_size (float): The proportion of the dataset to include in the validation split.
    strat (pd.Series or np.ndarray, optional): If not None, data is split in a stratified fashion, using this as the class labels.
    seed (int, optional): Random seed for reproducibility.
    
    Returns:
    tuple: The training and validation sets for features and target values.
    """
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=val_size, random_state=seed, stratify=strat)

    return X_train, X_val, y_train, y_val



def normalize(X_train, X_test):
    """
    Normalizes the data using the mean and standard deviation of the training set.
    
    Parameters:
    X_train (pd.DataFrame or np.ndarray): The training input samples.
    X_val (pd.DataFrame or np.ndarray): The validation input samples.
    
    Returns:
    tuple: Normalized training and validation sets.
    """
    # Normalize the data
    norm = Normalizer()
    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.transform(X_test)
    return X_train_norm, X_test_norm
