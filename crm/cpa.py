import logging
import multiprocessing as mp
from functools import lru_cache
from itertools import combinations
from typing import Union

import cartes
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import traffic
from cartopy import crs
from scipy.spatial import distance
from tqdm.notebook import tqdm

from crm import encounter_tools as et
from crm import los, tcas


@lru_cache
def project_cpa(
    traffic: traffic.core.Traffic,
    max_horizontal_spacing: float,
    max_vertical_spacing: float,
    projection: Union[pyproj.Proj, crs.Projection, None] = None,
    tau_mod=True,
    max_workers: int = 4,
) -> pd.DataFrame:
    logging.info("Retrieving close encounters")
    close_encounters = los.loss_of_spacing(
        traffic,
        max_horizontal_spacing=max_horizontal_spacing,
        max_vertical_spacing=max_vertical_spacing,
        projection=projection,
        max_workers=max_workers,
    )
    logging.info("Close encounters retrieved")
    #
    logging.info("Retrieving close traffic")
    close_traffic = los.keep_traffic_close_encounters(
        traffic, close_encounters, max_workers=max_workers
    )
    logging.info("Close traffic retrieved")
    logging.info("Retrieving simultaneous close traffic")
    df = et.simult_ids_from_traffic(
        close_traffic,
        agregate_speed_pos=True,
        projection=projection,
        max_workers=max_workers,
    )
    logging.info("simultaneous close traffic obtained")
    df["n"] = df["flight_id"].apply(len)
    df = df.query("n>1")
    df = df.drop(columns="n")

    for col in df.columns:
        df[col] = [[pair for pair in combinations(l, 2)] for l in df[col]]

    df = df.explode(list(df.columns))
    df2 = pd.DataFrame()
    for col in tqdm(df.columns, total=len(df.columns)):
        df2[[col + "_1", col + "_2"]] = pd.DataFrame(
            df[col].tolist(), index=df.index
        )
    df = df2

    df["rel_x"] = df["x_2"] - df["x_1"]
    df["rel_y"] = df["y_2"] - df["y_1"]
    df["rel_vx"] = df["SpeedX_2"] - df["SpeedX_1"]
    df["rel_vy"] = df["SpeedY_2"] - df["SpeedY_1"]
    df["rel_z"] = df["altitude_2"] - df["altitude_1"]
    df["rel_vz"] = df["vertical_rate_2"] - df["vertical_rate_1"]
    df["p_dot_p"] = df["rel_x"] * df["rel_x"] + df["rel_y"] * df["rel_y"]
    df["p_dot_v"] = df["rel_x"] * df["rel_vx"] + df["rel_y"] * df["rel_vy"] 
    df["v_dot_v"] = df["rel_vx"] * df["rel_vx"] + df["rel_vy"] * df["rel_vy"]
    df["is_converging"] = df["p_dot_v"] < 0
    df["horizontal_spacing"] = np.sqrt(df["p_dot_p"])
    df["time_to_cpa"] = -df["p_dot_v"] / df["v_dot_v"].replace(0, np.nan)
    df["x_1_at_cpa"] = df["x_1"] + df["time_to_cpa"] * df["SpeedX_1"]
    df["x_2_at_cpa"] = df["x_2"] + df["time_to_cpa"] * df["SpeedX_2"]
    df["y_1_at_cpa"] = df["y_1"] + df["time_to_cpa"] * df["SpeedY_1"]
    df["y_2_at_cpa"] = df["y_2"] + df["time_to_cpa"] * df["SpeedY_2"]
    df["altitude_1_at_cpa"] = (
        df["altitude_1"] + df["time_to_cpa"] * df["vertical_rate_1"] / 60
    )
    df["altitude_2_at_cpa"] = (
        df["altitude_2"] + df["time_to_cpa"] * df["vertical_rate_2"]/ 60
    )
    df["rel_x_at_cpa"] = df["x_2_at_cpa"] - df["x_1_at_cpa"]
    df["rel_y_at_cpa"] = df["y_2_at_cpa"] - df["y_1_at_cpa"]
    df["horizontal_spacing_at_cpa"] = np.sqrt(
        df["rel_x_at_cpa"] * df["rel_x_at_cpa"]
        + df["rel_y_at_cpa"] * df["rel_y_at_cpa"]
    )

    df["time_to_coalt"] = -df["rel_z"] / (
        df["rel_vz"].replace(0, np.nan) / 60
    )  # ft/s
    df["vertical_spacing_at_cpa"] = (
        df["rel_z"] + df["rel_vz"]/60 * df["time_to_cpa"]
    )

    if tau_mod:
        n_chunks = int(max_workers)
        df_chunks = np.array_split(df, n_chunks)
        with mp.Pool(max_workers) as pool:
            l_res = pool.map(
                _wrap_apply_taumod,
                tqdm(df_chunks, total=n_chunks, desc="adding tau mod"),
            )
        applied_df = pd.concat(l_res)
        df = pd.concat([df, applied_df], axis="columns")

    return df



def _wrap_apply_taumod(df):
    df = df.apply(
        lambda row: add_tau_mod_series(row),
        axis="columns",
        result_type="expand",
    )
    df.columns = ["tau_mod_ta", "tau_mod_ra"]
    return df


def add_tau_mod_series(row):
    sgz = np.max([row["altitude_1"], row["altitude_2"]])
    roc = np.max([row["vertical_rate_1"], row["vertical_rate_2"]])
    alt_agl = np.min([row["altitude_1"], row["altitude_2"]])  # TODO : use AGL

    tcas_lims = tcas.get_tcas_limits(sgz, roc, alt_agl)
    dmod_ta = tcas_lims.dmod_ta
    dmod_ra = tcas_lims.dmod_ra
    p_dot_v = row["p_dot_v"]
    if p_dot_v == 0:
        p_dot_v = np.nan
    tau_mod_ta = (dmod_ta ** 2 - row["p_dot_p"]) / p_dot_v
    tau_mod_ra = (dmod_ra ** 2 - row["p_dot_p"]) / p_dot_v
    return tau_mod_ta, tau_mod_ra

