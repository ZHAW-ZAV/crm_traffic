"""
Unit tests for the encounter tools functionalities.

"""
# pylint: disable=missing-function-docstring

import crm.encounter_tools as et
import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from traffic.core import Traffic

np.seterr(divide="ignore", invalid="ignore")


@pytest.fixture(scope="module")
def traffic():
    t1 = pd.date_range(start="1/1/2020", periods=5, freq="5S")
    t2 = pd.date_range(start="1/1/2020 00:00:10", periods=2, freq="5S")
    t3 = pd.date_range(start="1/1/2020 00:00:20", periods=2, freq="5S")
    times = t1.append(t2)
    times = times.append(t3)
    data = pd.DataFrame(
        {
            "flight_id": [
                "AAA",
                "AAA",
                "AAA",
                "AAA",
                "AAA",
                "BBB",
                "BBB",
                "CCC",
                "CCC",
            ],
            "callsign": [
                "AAA",
                "AAA",
                "AAA",
                "AAA",
                "AAA",
                "BBB",
                "BBB",
                "CCC",
                "CCC",
            ],
            "icao24": ["1", "1", "1", "1", "1", "2", "2", "3", "3"],
            "timestamp": times,
            "x": [0, 0, 0, 0, 0, 100, 200, 1000, 2000],
            "y": [0, 0, 0, 0, 0, 200, 100, 2000, 1000],
            "altitude": [0, 0, 0, 0, 0, 100, 150, 50, 10],
            "SpeedX": [0, 0, 0, 0, 0, -5, -4, 10, 100],
            "SpeedY": [0, 0, 0, 0, 0, -5, -5, 0, -10],
            "vertical_rate": [0, 0, 0, 0, 0, 1, 2, 10, 11],
        }
    )
    data.to_pickle("test_traffic.pkl")
    traffic = Traffic.from_file("test_traffic.pkl")
    return traffic


def test_simultaneous_traffic_retrieval(traffic):
    expected_result = pd.DataFrame(
        {
            "flight_id": [
                ["AAA"],
                ["AAA"],
                ["AAA", "BBB"],
                ["AAA", "BBB"],
                ["AAA", "CCC"],
                ["CCC"],
            ],
            "x": [[0], [0], [0, 100], [0, 200], [0, 1000], [2000]],
            "y": [[0], [0], [0, 200], [0, 100], [0, 2000], [1000]],
            "altitude": [[0], [0], [0, 100], [0, 150], [0, 50], [10]],
            "SpeedX": [[0], [0], [0, -5], [0, -4], [0, 10], [100]],
            "SpeedY": [[0], [0], [0, -5], [0, -5], [0, 0], [-10]],
            "vertical_rate": [[0], [0], [0, 1], [0, 2], [0, 10], [11]],
            "timestamp": pd.date_range(start="1/1/2020", periods=6, freq="5S"),
        }
    ).set_index("timestamp")
    simult_df = et.simult_ids_from_traffic(traffic, max_workers=1)
    assert_frame_equal(simult_df, expected_result)
