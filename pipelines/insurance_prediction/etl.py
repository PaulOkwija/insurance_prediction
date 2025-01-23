import pandas as pd
# from tabulate import tabulate

def extract_data(file_path):
    """
    Extracts data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The extracted data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    print(f"Extracted csv data from {file_path}. Shape is {data.shape}")
    return data

def transform_value(x):
    """
    Transforms specific string values to integers.

    Parameters:
    x (str): The value to transform.

    Returns:
    int: The transformed value.
    """
    if x == '   .':
        return 0
    elif x == '>=10':
        return 10
    else:
        return int(x)

def change_values_building(val):
    """
    Transforms building values.

    Parameters:
    val (str): The value to transform.

    Returns:
    int: 1 if val is 'N', otherwise 0.
    """
    if val == "N":
        return int(True)
    return int(False)

def change_values_garden(val):
    """
    Transforms garden values.

    Parameters:
    val (str): The value to transform.

    Returns:
    int: 1 if val is 'V', otherwise 0.
    """
    if val == "V":
        return int(True)
    return int(False)

def apply_transform(df, columns):
    """
    Applies the transform_value function to specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to transform.
    columns (list): The list of columns to apply the transformation to.

    Returns:
    pd.DataFrame: The transformed DataFrame.
    """
    for col in columns:
        df[col] = df[col].apply(transform_value)
    return df

def missing_val_removal(df, columns, method, rows=False):
    """
    Handles missing values in specified columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    columns (list): The list of columns to handle missing values for.
    method (str): The method to use for handling missing values ('median' or 'mode').
    rows (bool): Whether to drop rows with missing values. Default is False.

    Returns:
    pd.DataFrame: The DataFrame with missing values handled.
    """
    for col in columns:
        if method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])
    if rows:
        df = df.dropna()
    return df

def overview(dataframe):
    """
    Provides an overview of the DataFrame, including data types, missing values, and unique values per column.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to overview.

    Returns:
    None
    """
    print("\n ###################### Overview of data ###################### \n")

    # Viewing dtypes
    print("\n ### Datatypes ### \n")
    print(dataframe.dtypes)

    # Viewing missing values
    print("\n ### Null values ### \n")
    print(dataframe.isnull().sum())

    # Viewing number of unique values
    print("\n ### Unique values per column ### \n")
    print(dataframe.nunique())

    print("\n ###################### Overview Complete ###################### \n")



# def overview(dataframe):
#     """
#     Provides an overview of the DataFrame, including data types, missing values, and unique values per column.

#     Parameters:
#     dataframe (pd.DataFrame): The DataFrame to overview.

#     Returns:
#     None
#     """
#     print("\n###################### Overview of data ######################\n")

#     # Viewing dtypes
#     print("\n### Datatypes ###\n")
#     dtypes = dataframe.dtypes.reset_index()
#     dtypes.columns = ["Column", "Data Type"]
#     print(tabulate(dtypes, headers="keys", tablefmt="pretty"))

#     # Viewing missing values
#     print("\n### Null values ###\n")
#     null_values = dataframe.isnull().sum().reset_index()
#     null_values.columns = ["Column", "Null Values"]
#     print(tabulate(null_values, headers="keys", tablefmt="pretty"))
