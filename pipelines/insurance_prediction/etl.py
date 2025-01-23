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

def switch_to_int(df, columns):
    """
    Transforms specified columns in the dataframe by mapping values or creating dummy variables.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    columns (list of str): List of column names to be transformed.

    Returns:
    pandas.DataFrame: The transformed dataframe with specified columns mapped or converted to dummy variables.

    The function performs the following transformations:
    - For columns starting with 'Building', it maps the values using the `change_values_building` function.
    - For the column 'Garden', it maps the values using the `change_values_garden` function.
    - For all other columns, it converts them to dummy variables using `pd.get_dummies`.
    """
    for col in columns:
        if col.startswith('Building'):
            df[col] = df[col].map(change_values_building)
        elif col=='Garden':
            df[col] = df[col].map(change_values_garden)
        else:
          df = pd.get_dummies(df, columns=[col])
    return df

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

def preprocess(dataframe):
    """
    Preprocesses the DataFrame by cleaning data, transforming categorical data, handling missing values, and dropping unnecessary columns.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """

    print("\n ### Cleaning data.... ###\n")
    dataframe = apply_transform(dataframe, ['NumberOfWindows','Building_Type'])

    print("\n ### Swapping categorical data... ### \n")
    dataframe = switch_to_int(dataframe, ['Building_Painted','Building_Fenced','Garden','Settlement'])

    print("\n ### Dealing with missing values... ### \n")
    dataframe = missing_val_removal(dataframe,['Date_of_Occupancy','Building Dimension','Garden'],'median')

    print("\n ### Dropping Geo_code... ### \n")
    dataframe = dataframe.drop(['Geo_Code'],axis=1)
    print(f"Train shape is {dataframe.shape}")

    return dataframe

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
