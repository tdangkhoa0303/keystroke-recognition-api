import numpy as np
import pandas as pd

"""
sample = {
    "id": "",
    "events": [
        {
            "key": "A",
            "direction": 0,
            "timestamp": 1234567890,
        },
        {
            "key": "A",
            "direction": 1,
            "timestamp": 1234567891,
        },
        {
            "key": "B",
            "direction": 0,
            "timestamp": 1234567892,
        },
        {
            "key": "B",
            "direction": 1,
            "timestamp": 1234567893,
        },
        {
            "key": "C",
            "direction": 0,
            "timestamp": 1234567894,
        },
        {
            "key": "C",
            "direction": 1,
            "timestamp": 1234567895,
        },
    ]
}
"""


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
        curr_key = data[row_idx][1]
        curr_direction = data[row_idx][2]
        curr_timing = data[row_idx][3]

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
                press.append(curr_start)
                release.append(curr_end)

    resultant_data_frame = pd.DataFrame(
        list(zip(result_key, press, release)),
        columns=["Key", "Press_Time", "Release_Time"],
    )
    return resultant_data_frame


def get_DIG_features(data):

    result = [
        {
            "Keys": str(data[row_idx][0]) + "," + str(data[row_idx + 1][0]),
            "Holdtime1": (((data[row_idx][2] - data[row_idx][1])).microseconds) / 1000,
            "Holdtime2": (((data[row_idx + 1][2] - data[row_idx + 1][1])).microseconds)
            / 1000,
            "F1": (((data[row_idx + 1][1] - data[row_idx][2])).microseconds) / 1000,
            "F2": (((data[row_idx + 1][1] - data[row_idx][1])).microseconds) / 1000,
            "F3": (((data[row_idx + 1][2] - data[row_idx][2])).microseconds) / 1000,
            "F4": (((data[row_idx + 1][2] - data[row_idx][1])).microseconds) / 1000,
        }
        for row_idx in range(0, len(data))
        if (row_idx + 1 < len(data))
    ]

    df = pd.DataFrame(result)

    return df
