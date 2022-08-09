"""
Unit tests for the encounter tools functionalities.

"""
# pylint: disable=missing-function-docstring

import crm.los as los
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


@pytest.fixture(scope="module")
def row():
    return pd.Series(
        {
            "flight_id": ["AAA", "BBB", "CCC"],
            "x": [0, 200, -100],
            "y": [0, 100, 0],
            "altitude": [0, 150, -100],
            "SpeedX": [0, -4, 0],
            "SpeedY": [0, -5, 0],
            "vertical_rate": [0, 2, 0],
        }
    )


def test_get_horiz_spacing(row):
    horiz_spacing = los.get_horiz_spacing(row)
    d12 = (200 ** 2 + 100 ** 2) ** 0.5
    d13 = (100 ** 2 + 0 ** 2) ** 0.5
    d23 = (300 ** 2 + 100 ** 2) ** 0.5
    expected_result = np.array(
        [[np.nan, d12, d13], [d12, np.nan, d23], [d13, d23, np.nan]]
    )
    np.testing.assert_allclose(
        horiz_spacing, expected_result, rtol=0.1, atol=0.1
    )


def test_get_vert_spacing(row):
    vert_spacing = los.get_vert_spacing(row)
    expected_result = np.array(
        [[np.nan, 150, 100], [150, np.nan, 250], [100, 250, np.nan]]
    )
    np.testing.assert_allclose(
        vert_spacing, expected_result, rtol=0.1, atol=0.1
    )


def test_get_loss_of_spacing(row):
    is_los = los.get_loss_of_spacing(row, horiz_lim=230, vert_lim=300)
    expected_result = np.array(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    np.testing.assert_allclose(is_los, expected_result)

    is_los = los.get_loss_of_spacing(row, horiz_lim=230, vert_lim=120)
    expected_result = np.array(
        [[False, False, True], [False, False, False], [True, False, False]]
    )
    np.testing.assert_allclose(is_los, expected_result)


def test_get_loss_of_spacing_series(row):
    tple_los = los.get_loss_of_spacing_series(row, horiz_lim=230, vert_lim=300)
    expected_result = (
        [np.array(["AAA", "BBB"]), np.array(["AAA", "CCC"])],
        [223.60679775, 100],
        [150, 100],
    )
    np.testing.assert_array_equal(expected_result[0][0], tple_los[0][0])
    np.testing.assert_array_equal(expected_result[0][1], tple_los[0][1])
    np.testing.assert_almost_equal(tple_los[1], expected_result[1])
    np.testing.assert_almost_equal(tple_los[2], expected_result[2])

    tple_los = los.get_loss_of_spacing_series(row, horiz_lim=230, vert_lim=120)
    expected_result = (
        [np.array(["AAA", "CCC"])],
        [100],
        [100],
    )
    np.testing.assert_array_equal(expected_result[0][0], tple_los[0][0])
    np.testing.assert_almost_equal(tple_los[1], expected_result[1])
    np.testing.assert_almost_equal(tple_los[2], expected_result[2])
