# Train Keras prediction models for time series

import glob
import os
import pandas as pd
import numpy as np
import keras
import gc

# Read cached timeseries from the Citadel API into Pandas dataframes
def read_timeseries_dfs(target_uuid, data_dir='data/'):
    files = sorted(glob.glob(data_dir + '*.json'))
    target_df = None
    dfs = []

    for idx, filepath in enumerate(files):
        if idx % 10 == 0:
            print(idx, end=' ', flush=True)

        uuid = os.path.splitext(os.path.basename(filepath))[0]

        f = open(filepath, 'r')
        df = pd.read_json(f)
        f.close()

        df = df.rename(columns={'data': uuid})

        if len(df) == 0:
            continue
        if df[uuid].std() < 1e-10:
            continue

        if uuid == target_uuid:
            target_df = df
        else:
            dfs.append(df)

    print()
    assert(target_df is not None)
    return (target_df, dfs)

# Processing steps include normalizing each feature by the time series mean
# and standard deviation, and finding the closest values to each reading in
# the target time series.
def process_dfs(target_df, dfs):
    processed_dfs = []
    for idx, df in enumerate(dfs):
        if idx % 100 == 0:
            print(idx, end=' ', flush=True)

        uuid = df.columns[0]
        df[uuid] = (df[uuid] - df[uuid].mean()) / (df[uuid].std())

        # Find indices in the current feature df which are closest to each
        # reading in the target_df
        indices = np.searchsorted(df.index, target_df.index)
        indices[indices == len(df)] = len(df) - 1
        target_df['indices'] = df.index.get_values()[indices]

        # Extract the corresponding sensor value for each reading in the
        # target_df
        merged = pd.merge(target_df, df, left_on='indices', right_index=True)
        processed_dfs.append(merged[[uuid]])

    print()
    target_df.drop('indices', axis=1, inplace=True)
    return processed_dfs

# Add one-hot feature columns for certain time features.
def add_time_features(all_data):
    months     = pd.get_dummies(all_data.index.month, prefix='month')
    daysofweek = pd.get_dummies(all_data.index.dayofweek, prefix='dayofweek')
    hours      = pd.get_dummies(all_data.index.hour, prefix='hour')

    months.index     = all_data.index
    daysofweek.index = all_data.index
    hours.index      = all_data.index

    all_data = pd.concat([all_data, months, daysofweek, hours], axis=1)
    return all_data

def train_model(target_uuid, num_hidden_units, num_autoregressive_terms,
                delay_length, num_output_values):
    target_df, dfs = read_timeseries_dfs(target_uuid)
    processed_dfs = process_dfs(target_df, dfs)

    # Add autoregressive and time features
    all_data = pd.concat([target_df] + processed_dfs, axis=1)
    feature_names = list(all_data.columns)
    for i in range(1, num_autoregressive_terms + 1):
        all_data['prev_value_' + str(i)] = all_data[target_uuid].shift(i + delay_length)
    all_data = all_data.iloc[(num_autoregressive_terms + delay_length):]
    for i in range(1, num_autoregressive_terms + 1):
        colname = 'prev_value_' + str(i)
        all_data[colname] = (all_data[colname] - all_data[colname].mean()) / all_data[colname].std()
    all_data = add_time_features(all_data)

    # Add future output terms
    target_columns = [target_uuid]
    for i in range(1, num_output_values):
        all_data['future_value_' + str(i)] = all_data[target_uuid].shift(-i)
        target_columns.append('future_value_' + str(i))
    if num_output_values > 1:
        all_data = all_data.iloc[:-num_output_values+1]

    # Create training and test data
    shuffled_data = all_data.sample(frac=1, random_state=1)
    df_train = shuffled_data.iloc[:int(0.8*len(shuffled_data))]
    df_test  = shuffled_data.iloc[int(0.8*len(shuffled_data)):]

    x_train = df_train.drop(target_columns, axis=1).values
    y_train = df_train[target_columns].values.reshape([len(df_train), num_output_values])
    x_test = df_test.drop(target_columns, axis=1).values
    y_test = df_test[target_columns].values.reshape([len(df_test), num_output_values])

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(num_hidden_units, activation='relu',
                                 input_shape=(x_train.shape[1],)))
    model.add(keras.layers.Dense(num_output_values, activation='linear'))
    model.summary()

    starting_lr = target_df[target_uuid].mean() / 1e5
    rmsprop = keras.optimizers.RMSprop(lr=starting_lr)
    model.compile(loss='mse', optimizer=rmsprop)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        patience=5, verbose=1)
    model.fit(x_train, y_train, batch_size=64, epochs=400,
        verbose=1, validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr])

    # Save model
    model.save('models/' + target_uuid)

    return all_data.shape, feature_names

def get_forecast(model_info):
    target_df, dfs = read_timeseries_dfs(model_info['target_uuid'])
    processed_dfs = process_dfs(target_df, dfs)

    model = keras.models.load_model('models/' + model_info['target_uuid'])

    # Add autoregressive and time features
    all_data = pd.concat([target_df] + processed_dfs, axis=1)
    for i in range(1, model_info['num_autoregressive_terms'] + 1):
        all_data['prev_value_' + str(i)] = all_data[model_info['target_uuid']].shift(i + model_info['delay_length'])
    all_data = all_data.iloc[(model_info['num_autoregressive_terms'] + model_info['delay_length']):]
    for i in range(1, model_info['num_autoregressive_terms'] + 1):
        colname = 'prev_value_' + str(i)
        all_data[colname] = (all_data[colname] - all_data[colname].mean()) / all_data[colname].std()

    all_data = add_time_features(all_data)

    # Extract the most recent point in the dataset to obtain predictions
    all_data = all_data.drop(model_info['target_uuid'], axis=1)
    test_point = all_data.iloc[[-1]].values

    # Make and return predictions on the last point in the dataset - which
    # corresponds to forecasting sensor values in the future. The number of
    # predictions depends on the number of output values specified in the model.
    predictions = model.predict(test_point)
    predictions = list(predictions.flatten())
    predictions = list(map(float, predictions))

    return predictions

def get_anomalies(model_info):
    target_df, dfs = read_timeseries_dfs(model_info['target_uuid'])
    processed_dfs = process_dfs(target_df, dfs)
    model = keras.models.load_model('models/' + model_info['target_uuid'])

    # Add autoregressive and time features
    all_data = pd.concat([target_df] + processed_dfs, axis=1)
    for i in range(1, model_info['num_autoregressive_terms'] + 1):
        all_data['prev_value_' + str(i)] = all_data[model_info['target_uuid']].shift(i + model_info['delay_length'])
    all_data = all_data.iloc[(model_info['num_autoregressive_terms'] + model_info['delay_length']):]
    for i in range(1, model_info['num_autoregressive_terms'] + 1):
        colname = 'prev_value_' + str(i)
        all_data[colname] = (all_data[colname] - all_data[colname].mean()) / all_data[colname].std()
    all_data = add_time_features(all_data)

    all_x = all_data.drop(model_info['target_uuid'], axis=1).values
    all_y = all_data[model_info['target_uuid']].values.reshape([len(all_data), 1])

    all_data['prediction'] = model.predict(all_x)[:,0]
    errors = all_data[model_info['target_uuid']] - all_data['prediction']

    # Anomalies are errors which are outside +/- 3 standard deviations from the
    # mean error value
    mean = np.mean(errors)
    std = np.std(errors)
    max_cutoff = mean + 3*std
    min_cutoff = mean - 3*std

    anomalies_x = []
    anomalies_y = []
    for i, error in enumerate(errors):
        if error > max_cutoff or error < min_cutoff:
            anomalies_x.append(errors.index[i])
            anomalies_y.append(error)

    anomalies_x = [int(t.timestamp()) for t in anomalies_x]

    return (anomalies_x, anomalies_y)

def get_correlations(model_info):
    target_df, dfs = read_timeseries_dfs(model_info['target_uuid'])
    processed_dfs = process_dfs(target_df, dfs)
    model = keras.models.load_model('models/' + model_info['target_uuid'])

    # Add autoregressive and time features
    all_data = pd.concat([target_df] + processed_dfs, axis=1)
    for i in range(1, model_info['num_autoregressive_terms'] + 1):
        all_data['prev_value_' + str(i)] = all_data[model_info['target_uuid']].shift(i + model_info['delay_length'])
    all_data = all_data.iloc[(model_info['num_autoregressive_terms'] + model_info['delay_length']):]
    for i in range(1, model_info['num_autoregressive_terms'] + 1):
        colname = 'prev_value_' + str(i)
        all_data[colname] = (all_data[colname] - all_data[colname].mean()) / all_data[colname].std()
    all_data = add_time_features(all_data)

    # Add future output terms
    target_columns = [model_info['target_uuid']]
    for i in range(1, model_info['num_output_values']):
        all_data['future_value_' + str(i)] = all_data[model_info['target_uuid']].shift(-i)
        target_columns.append('future_value_' + str(i))
    if model_info['num_output_values'] > 1:
        all_data = all_data.iloc[:-model_info['num_output_values']+1]

    column_results = []
    for i, column in enumerate(all_data.columns):
        print(i, end=' ', flush=True)
        print(column)
        if column in target_columns:
            continue

        all_data_copy = all_data.copy()
        all_data_copy[column] = 0
        x = all_data_copy.drop(target_columns, axis=1).values
        y = all_data_copy[target_columns].values.reshape([len(all_data), model_info['num_output_values']])

        rmse = np.sqrt(model.evaluate(x, y, verbose=0, batch_size=len(all_data)))
        column_results.append((column, rmse))

        del all_data_copy
        del x
        del y
        gc.collect()

    print()
    column_results.sort(key=lambda x: x[1], reverse=True)
    return column_results
