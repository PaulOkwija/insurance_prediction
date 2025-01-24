from insurance_prediction import *

def run_etl(file_path):
    """
    Orchestrates the ETL process.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    data = extract_data(file_path)
    features = data.iloc[:,1:-1]
    labels = data.iloc[:,-1]
    preprocessed_features = preprocess(features)
    overview(preprocessed_features)
    return preprocessed_features, labels



def run_pipeline(file_path, val_size=0.2, feature_selection=True):
    """
    Orchestrates the entire pipeline.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    preprocessed_features, labels = run_etl(file_path)
    X_train, X_test, y_train, y_test = split_data(preprocessed_features, labels, val_size, strat=labels, seed=42)

    if feature_selection:
        features_bool = feature_select(X_train, y_train)
        X_train = X_train[:, features_bool]
        X_test = X_test[:, features_bool]


    model = train_model(preprocessed_features, labels)
    y_test_pred = model.predict(X_test)
    evaluate_model(y_test, y_test_pred, category="Test")
    plot_confusion_matrix(y_test, y_test_pred, [0, 1], title="Test Confusion matrix")
    return model

if __name__ == "__main__":
    run_pipeline("/content/insurance_prediction/data/0_raw/train_data.csv")