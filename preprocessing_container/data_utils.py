import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Optional
from sklearn.impute import KNNImputer # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # type: ignore

def read_data(data_path: str, filter: Optional[str] = None) -> pd.DataFrame:
    '''
    Reads a csv file with optional filtering by identifier
    '''
    df = pd.read_csv(data_path)

    try:
        df['time'] = pd.to_datetime(df['time'])
    except:
        raise ValueError('No column labeled \'time\'')

    # Selects only data for user-defined identifer
    if filter is not None:
        df = df.loc[df['District'] == filter]
        df.drop(['District'], axis=1, inplace=True)
        if isinstance(df, pd.Series):
            df = df.to_frame().T  # Ensure return type is always DataFrame
    
    # Index is set to the time so it doesn't clog other functions that can only operate on numerical data types
    try:
        df.set_index(pd.DatetimeIndex(df['time']), inplace=True)
        df.drop(['time'], axis=1, inplace=True)
    except:
        raise Exception("Error: Could not set 'time' column as index")

    # Casts all numeric values to float64 for minimal data loss in the case of high precision values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float64)

    return df

def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().select_dtypes(include=np.number)

def handle_nans(df: pd.DataFrame, threshold: float = 0.33, window: int = 2, no_drop: bool = False) -> pd.DataFrame:
    """
    Handle NaN values in the DaSaFrame by dropping rows with too many NaNs.
    Threshold is the highest percentage of NaNs allowed in a row.
    Rows with more than this percentage of NaNs will be dropped.
    Remaining NaNs will be filled by KNN imputation.
    Window is the number of nearby non-NaN values used in imputation.
    """
    df_return = select_numeric(df)
    features = df_return.columns

    # Drop rows with too many NaNs
    if not no_drop:
        thresh_not_missing = np.ceil(len(features) * (1 - threshold))
        df_return.dropna(axis=0, thresh=thresh_not_missing, inplace=True)

    # Drop columns with only NaNs
    df_return.dropna(axis=1, how='all', inplace=True)

    # Impute missing values via KNNImputer (sci-kit learn)
    for col in features:
        imputer = KNNImputer(n_neighbors=window, weights='distance', copy=True)
        array_col = df_return[col].to_numpy().reshape(-1, 1)
        preserve_index = df_return[col].index
        df_return[col] = pd.Series(imputer.fit_transform(array_col).flatten(), index=preserve_index)

    return df_return

def scale_data(df: pd.DataFrame, scale: str='StandardScaler') -> pd.DataFrame:
    df_return = select_numeric(df)

    match scale:
        case 'StandardScaler':
            scaler = StandardScaler()
        case 'MinMaxScaler':
            scaler = MinMaxScaler()
        case 'RobustScaler':
            scaler = RobustScaler()
        case 'MaxAbsScaler':
            scaler = MaxAbsScaler()
        case _:
            raise ValueError('Argument passed is not a supported scaler')
        
    df_return = pd.DataFrame(scaler.fit_transform(df_return), index=df.index, columns=df.columns)

    return df_return

def generate_lags(df: pd.DataFrame, n_lags: int, step: int=1) -> pd.DataFrame:
    '''
    Generates duplicate n_lags duplicate features that are lagged by step
    '''
    df_n = df.copy()
    columns = df.columns
    sampling = check_uniform(df)
    dfs = []

    for n in range(step, step*n_lags + 1, step):
        for col in columns:
            df_n[f"{col}_lag(-{n})"] = df_n[col].shift(n, freq=sampling)
    df_n = df_n.iloc[n_lags:]

    dfs.append(df_n)
    df_return = pd.concat(dfs, ignore_index=False)

    return df_return

def check_uniform(df: pd.DataFrame) -> pd.Timedelta:
    '''
    Assuming a dataframe with a datetime index, this function checks for uniformity
    in sampling and returns the most common Timedelta
    '''

    # Ensure the index is a DatetimeIndex for reliable Timedelta operations
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    
    time_diffs: pd.Series = df.index.to_series().diff()
    time_diffs_mode = time_diffs.mode()

    if not time_diffs_mode.empty:
        # Explicitly extract the first element and cast it to pd.Timedelta.
        most_common_frequency = pd.Timedelta(time_diffs_mode.iloc[0])
    else:
        # If there's less than 2 data points, diff() will be all NaT or empty,
        # leading to an empty mode.
        raise ValueError('Cannot determine most common frequency, not enough valid time differences.')

    tolerance = pd.Timedelta(milliseconds=10)
    time_diffs[abs(time_diffs - most_common_frequency) <= tolerance] = most_common_frequency

    diff_counts = time_diffs.value_counts().sort_index()

    if diff_counts.empty:
        raise ValueError('No valid time differences found to calculate frequency.')

    total_observations = len(time_diffs)
    count_most_common = diff_counts.loc[most_common_frequency]
    percentage_most_common = (count_most_common / total_observations) * 100
        
    if percentage_most_common < 75:
        print(f"Most common frequency accounts for {percentage_most_common:.2f}% of the time steps.")
        print('Warning: sampling frequency is highly irregular. Resampling is strongly recommended')
    elif percentage_most_common < 90:
        print(f"Most common frequency accounts for {percentage_most_common:.2f}% of the time steps.")
        print('Warning: sampling frequency is irregular. Resampling is recommended')

    return most_common_frequency

def time_to_feature(df: pd.DataFrame):
    '''
    Creates features for cyclical representations of time to help ML models identify seasonality.
    Cyclic representations are necessary to show that 23:59 is very close to 0:00
    '''
    df_return = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    

    df_return = (
        df_return
        .assign(minute=(df.index.hour*60 + df.index.minute))
        .assign(dayofweek=df.index.dayofweek)
        .assign(dayofyear=df.index.dayofyear)
    )

    time_features = {'minute': 1440, 'dayofweek': 7, 'dayofyear': 365.25}

    for feature, period in time_features.items():
        df_return[f"sin_{feature}"] = np.sin((df_return[feature]) * (2 * np.pi / period))
        df_return[f"cos_{feature}"] = np.cos((df_return[feature]) * (2 * np.pi / period))
        df_return = df_return.drop(columns=[feature])

    return df_return


