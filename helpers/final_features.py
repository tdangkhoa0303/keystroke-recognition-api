import pandas as pd
import numpy as np


def get_timings_KIT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)
    for i, (key, timing) in enumerate(keys_in_pipeline):
        if search_key == key:
            mask[i] = 0
            non_zero_indices = np.nonzero(mask)

            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []

            return keys_in_pipeline, timing, search_key_timing
    return keys_in_pipeline, None, None


# function to get KIT data frame with key, press_time, release_time for a given user
def get_dataframe_KIT(data):
    keys_in_pipeline = []
    result_key = []
    press = []
    release = []
    for row_idx in range(len(data)):
        keys_in_pipeline = list(keys_in_pipeline)
        curr_key = data[row_idx][0]
        curr_direction = data[row_idx][1]
        curr_timing = data[row_idx][2]

        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

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


def get_DIG_features(data):
    """
    - Flight1 Ki-Ki+1 : Ki+1.Press - Ki.Release
    - Flight2 Ki-Ki+1 : Ki+1.Release - Ki.Release
    - Flight3 Ki-Ki+1 : Ki+1.Press - Ki.Press
    - Flight4 Ki-Ki+1 : Ki+1.Release - Ki.Press
    """
    result = [
        {
            "Keys": str(data[row_idx][0]) + "," + str(data[row_idx + 1][0]),
            "Holdtime1": data[row_idx][2] - data[row_idx][1],
            "F1": data[row_idx + 1][1] - data[row_idx][2],
            "F2": data[row_idx + 1][2] - data[row_idx][2],
            "F3": data[row_idx + 1][1] - data[row_idx][1],
            "F4": data[row_idx + 1][2] - data[row_idx][1],
        }
        for row_idx in range(0, len(data))
        if (row_idx + 1 < len(data))
    ]

    df = pd.DataFrame(result)

    return df


def get_words(events):
    words_in_pipeline = []
    word_list = []

    delimiter_keys = ["SPACE", ".", ",", "RETURN", " "]
    up_events = filter(lambda event: event["direction"] == 1, events)

    for event in up_events:
        curr_key = event["key"]  # Adjust to access keyName

        if curr_key in delimiter_keys:
            # If there's a current word being formed, finalize it
            if len(words_in_pipeline) > 0:
                # Join the characters in words_in_pipeline to form a word
                key_word = "".join(words_in_pipeline)
                word_list.append(key_word)
                # Clear the words_in_pipeline for the next word
                words_in_pipeline = []
            continue

        if curr_key.upper() == "BACKSPACE":
            # Remove the last character if it's not empty
            if words_in_pipeline:
                words_in_pipeline.pop()  # Remove the last character
            continue

        # Append the current key to the words_in_pipeline
        words_in_pipeline.append(curr_key)

    # Handle the case where the last word may not end with a delimiter
    if words_in_pipeline:
        key_word = "".join(words_in_pipeline)
        word_list.append(key_word)

    return word_list


def compute_features(df):
    kit_data = get_dataframe_KIT(df.values)
    dig_data = get_DIG_features(kit_data.values)

    # CapsLock Usage
    capslock_usage = df[df["key"] == "CAPSLOCK"].shape[0]

    number_of_negative_UD = sum(dig_data["F1"] < 0)
    total_pairs = len(dig_data)
    number_of_negative_UU = sum(dig_data["F2"] < 0)
    neg_ud_percentage = number_of_negative_UD / total_pairs
    neg_uu_percentage = number_of_negative_UU / total_pairs

    # Shift Ratios (RSA & LSA)
    right_shift_usage = df[df["key"] == "RSHIFT"].shape[0]  # Assuming 16 is right shift
    left_shift_usage = df[df["key"] == "LSHIFT"].shape[0]  # Assuming 42 is left shift
    total_shift_usage = right_shift_usage + left_shift_usage
    rsa_ratio = right_shift_usage / total_shift_usage if total_shift_usage != 0 else 0
    lsa_ratio = left_shift_usage / total_shift_usage if total_shift_usage != 0 else 0

    # Return all metrics in a dictionary
    return {
        "capsLock_usage": capslock_usage,
        "rsa_ratio": rsa_ratio,
        "lsa_ratio": lsa_ratio,
        "negative_ud": neg_ud_percentage,
        "negative_uu": neg_uu_percentage,
        "mean_hold_time1": dig_data["Holdtime1"].mean() / 1000,
        "mean_f1": dig_data["F1"].mean() / 1000,
        "mean_f2": dig_data["F2"].mean() / 1000,
        "mean_f3": dig_data["F3"].mean() / 1000,
        "mean_f4": dig_data["F4"].mean() / 1000,
    }
