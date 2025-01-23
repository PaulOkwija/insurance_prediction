from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def feature_select(x_train, y_train):
    """
    Selects important features from the training data using a Random Forest classifier.
    
    Parameters:
    x_train (pd.DataFrame or np.ndarray): The training input samples.
    y_train (pd.Series or np.ndarray): The target values.
    
    Returns:
    np.ndarray: A boolean array indicating which features are selected.
    """
    # Using random forest to fit data
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
    rf.fit(x_train, y_train)
    print("Rf score", rf.score(x_train, y_train))

    model = SelectFromModel(rf, prefit=True)
    features_bool = model.get_support()
    # features = train_features.columns[features_bool]
    return features_bool

def split_data(train_features, train_y, val_size, strat=None, seed=42):
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
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_y, test_size=val_size, random_state=seed, stratify=strat)

    return X_train, X_val, y_train, y_val