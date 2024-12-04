import pandas as pd

# Reverse mapping for raw data keys to descriptive keys
RAW_KEY_MAPPING = {
    '.': 'period',
    '5': 'five'
}

# Function to reverse map the raw key to the descriptive key
def reverse_map_key(key):
    return RAW_KEY_MAPPING.get(key, key)

# Assuming get_timings_KIT is implemented elsewhere
def get_timings_KIT(keys_in_pipeline, curr_key, curr_timing):
    for idx, entry in enumerate(keys_in_pipeline):
        if entry['key'] == curr_key:
            keys_in_pipeline.pop(idx)
            return keys_in_pipeline, entry['timestamp'], curr_timing
    return keys_in_pipeline, None, None

def get_dataframe_KIT(data):
    keys_in_pipeline = []
    result_key = []
    press = []
    release = []
    for row_idx in range(len(data)):
        curr_key = reverse_map_key(data[row_idx]['key'])
        curr_direction = data[row_idx]['direction']
        curr_timing = data[row_idx]['timestamp'] / 1000

        if curr_direction == 0:
            keys_in_pipeline.append({'key': curr_key, 'timestamp': curr_timing})

        if curr_direction == 1:
            keys_in_pipeline, curr_start, curr_end = get_timings_KIT(
                keys_in_pipeline, curr_key, curr_timing
            )
            if curr_start is None:
                continue
            else:
                result_key.append(curr_key)
                press.append(float(curr_start))
                release.append(curr_end)

    resultant_data_frame = pd.DataFrame(
        list(zip(result_key, press, release)),
        columns=["Key", "Press_Time", "Release_Time"],
    )
    return resultant_data_frame

def extract_keystroke_features(data, key_sequence):
    # Preprocess the raw data into a structured DataFrame
    df = get_dataframe_KIT(data)

    # Initialize the features dictionary
    features = {f'H.{key}': 0 for key in key_sequence}
    features.update({f'DD.{key1}.{key2}': 0 for key1, key2 in zip(key_sequence, key_sequence[1:])})
    features.update({f'UD.{key1}.{key2}': 0 for key1, key2 in zip(key_sequence, key_sequence[1:])})

    # Calculate Hold Time (H)
    for key in key_sequence:
        key_data = df[df['Key'] == key]
        if not key_data.empty:
            features[f'H.{key}'] = key_data.iloc[0]['Release_Time'] - key_data.iloc[0]['Press_Time']

    # Calculate Down-Down Time (DD) and Up-Down Time (UD)
    for key1, key2 in zip(key_sequence, key_sequence[1:]):
        key1_data = df[df['Key'] == key1]
        key2_data = df[df['Key'] == key2]
        if not key1_data.empty and not key2_data.empty:
            dd = key2_data.iloc[0]['Press_Time'] - key1_data.iloc[0]['Press_Time']
            ud = key2_data.iloc[0]['Press_Time'] - key1_data.iloc[0]['Release_Time']
            features[f'DD.{key1}.{key2}'] = dd
            features[f'UD.{key1}.{key2}'] = ud

    return features
