import logging
import multiprocessing as mp
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyproj
from cartopy import crs
from tqdm.notebook import tqdm
from traffic.core import Traffic


@lru_cache
def simult_ids_from_traffic(
    traffic: Traffic,
    agregate_speed_pos: bool = True,
    projection: Union[pyproj.Proj, crs.Projection, None] = None,
    max_workers: int = 4,
    extra_cols2agregate: Optional[tuple] = None,
) -> pd.DataFrame:
    """Return timeseries of flight ids seen at the same time.

    Parameters
    ----------
    traffic : Traffic
        Traffic object which is a collection of flights.
    agregate_speed_pos : bool, optional
        Add position and speed infromation columns by default True.
    projection : Union[pyproj.Proj, cartes.crs.Projection, None], optional
        You need to provide a decentprojection able to approximate distances by
        Euclide formula. By default, EuroPP() projection is considered, but a
        non explicit argument will raise a warning., by default None
    max_workers : int, optional
        Number of cpu cores to use, by default 4.
    extra_cols2agregate: list optional
        Tuple of column names that need to be aggregated.

    Returns
    -------
    pd.DataFrame
        Timeseries with a column of flight ids for flight occuring at each
        timestamp. If agregate_speed_pos is True it also contains columns for
        speeds and positions. The speed and positions are in SI units (m and
        m.s-1).
    """
    if agregate_speed_pos:
        if projection is None:
            logging.warning("Defaulting to projection EuroPP()")
            projection = crs.EuroPP()
        if ("x" not in traffic.data.columns) or (
            "y" not in traffic.data.columns
        ):
            traffic = traffic.compute_xy(projection)

    if max_workers == 1:
        simult_ids = _wrap_groupby_agg(
            *(
                traffic.data.sort_values("timestamp"),
                "timestamp",
                agregate_speed_pos,
                extra_cols2agregate,
            )
        )
        simult_ids_sorted =  simult_ids.apply(
            lambda row: [list(t) for t in zip(*sorted(zip(*row)))],
            axis=1,
            result_type="expand"
            )
        simult_ids_sorted.columns = simult_ids.columns
        return simult_ids_sorted

    n_chunks = int(max_workers)
    df_chunks = np.array_split(traffic.data.sort_values("timestamp"), n_chunks)
    chunks = [
        (df, "timestamp", agregate_speed_pos, extra_cols2agregate)
        for df in df_chunks
    ]

    with mp.Pool(max_workers) as pool:
        l_res = pool.starmap(
            _wrap_groupby_agg,
            tqdm(chunks, total=n_chunks, desc="simultaneous flights"),
        )
    simult_ids = pd.concat(l_res).sort_index()
    
    return simult_ids
    # TODO : Fix missing due to bad splitting
    # dup_ts = simult_ids.index[simult_ids.index.duplicated(keep=False)]
    # simult_ids.loc[dup_ts] = simult_ids.loc[dup_ts].groupby("timestamp").sum()
    # return simult_ids.loc[~simult_ids.index.duplicated(keep="first")]


def _wrap_groupby_agg(
    df: pd.DataFrame,
    col2group: str,
    agregate_speed_pos: bool,
    extra_cols2agregate: Optional[tuple] = None,
):
    if agregate_speed_pos:
        cols2get = {
            "flight_id": list,
            "x": list,
            "y": list,
            "altitude": list,
            "SpeedX": list,
            "SpeedY": list,
            "vertical_rate": list,
            "track": list,
        }

        knts2ms = 0.514444
        if "SpeedX" not in df.columns:
            df["SpeedX"] = (
                np.sin(np.deg2rad(df["track"])) * df["groundspeed"] * knts2ms
            )
        if "SpeedY" not in df.columns:
            df["SpeedY"] = (
                np.cos(np.deg2rad(df["track"])) * df["groundspeed"] * knts2ms
            )
    else:
        cols2get = {"flight_id": list}

    if extra_cols2agregate is not None:
        for cols in extra_cols2agregate:
            cols2get[cols] = list

    simult_ids =  df.groupby(col2group).agg(cols2get)
    simult_ids_sorted =  simult_ids.apply(
        lambda row: [list(t) for t in zip(*sorted(zip(*row)))],
        axis=1,
        result_type="expand"
        )
    simult_ids_sorted.columns = simult_ids.columns
    return simult_ids
