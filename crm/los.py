import logging
import multiprocessing as mp
from functools import lru_cache
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
from traffic.core import Traffic

from crm import encounter_tools as et

DICO_FILTERING: dict

def get_horiz_spacing(row: pd.Series) -> npt.ArrayLike:
    """Compute the horizontal spacing between x,y coordinates.

    Parameters
    ----------
    row : pd.Series
        Pandas row containing x and y coordinates

    Returns
    -------
    np.array
        pairwise euclidian distances
    """
    cols = ["x", "y"]
    coords = [(x, y) for x, y in zip(row[cols].values[0], row[cols].values[1])]

    horiz_spacing = distance.cdist(coords, coords, "euclidean")
    np.fill_diagonal(horiz_spacing, np.nan)
    return horiz_spacing


def get_vert_spacing(row: pd.Series) -> npt.ArrayLike:
    """Compute the vertical spacing between x,y coordinates.

    Parameters
    ----------
    row : pd.Series
        Pandas row containing altitudes

    Returns
    -------
    np.array
        pairwise euclidian distances
    """
    cols = ["altitude"]
    coords = np.array(row[cols].values[0]).reshape(-1, 1)
    vert_spacing = distance.cdist(coords, coords, "euclidean")
    np.fill_diagonal(vert_spacing, np.nan)
    return vert_spacing


def add_spacing(row: pd.Series):
    """Add horizontal and vertical spacing to pandas series"""
    row["vert_spacing"] = get_vert_spacing(row)
    row["horiz_spacing"] = get_horiz_spacing(row)
    return row


def get_loss_of_spacing(
    row: pd.Series, horiz_lim: float, vert_lim: float
) -> npt.ArrayLike:
    """Compute where loss of spacing is happening.

    A loss of spacing is happeing when both a vertical and an horizontal
    maximum allowed spacing are violated.

    Parameters
    ----------
    row : pd.Series
        Series containing horiz_spacing and vert_spacing columns.
    horiz_lim : float
        Maximum horizontal spacing allowed.
    vert_lim : float
        Maximum vertical spacing allowed.

    Returns
    -------
    los : np.array
        Pairwise array containing True where loss of spacing is found and False
        otherwise.
    """
    if "horiz_spacing" not in row:
        row["horiz_spacing"] = get_horiz_spacing(row)
    if "vert_spacing" not in row:
        row["vert_spacing"] = get_vert_spacing(row)
    los = (row.horiz_spacing < horiz_lim) & (row.vert_spacing < vert_lim)
    return los


def get_loss_of_spacing_series(
    row: pd.Series, horiz_lim: float, vert_lim: float
) -> tuple:
    """Return pairs having a loss of spacing and their corresponding spacing.

    Parameters
    ----------
    row : pd.Series
        Series containing the column flight_id, x and y
    horiz_lim : float
        Maximum horizontal spacing allowed.
    vert_lim : float
        Maximum vertical spacing allowed.

    Returns
    -------
    tuple
        (pairs, horiz_spacings, vert_spacing)
    """
    los = get_loss_of_spacing(row, horiz_lim, vert_lim)
    where_los = np.where(los)
    indexes = list(
        set([tuple(sorted([a, b])) for a, b in zip(*np.where(los))])
    )
    # Sorting pairs will improve latter usage
    pairs = [sorted(np.array(row["flight_id"])[[a, b]]) for a, b in indexes]
    horiz_spacings = [row["horiz_spacing"][a, b] for a, b in indexes]
    vert_spacings = [row["vert_spacing"][a, b] for a, b in indexes]

    if "NSE" in row.index:
        # TODO sort NSE with flight IDs
        nse = [sorted(np.array(row["NSE"])[[a, b]]) for a, b in indexes]
        return (pairs, horiz_spacings, vert_spacings, nse)
    else:
        return (
            pairs,
            horiz_spacings,
            vert_spacings,
        )


def get_loss_of_spacing_df(
    sim_ids: pd.DataFrame, horiz_lim: float, vert_lim: float
) -> pd.DataFrame:
    """Take a simultaneous df and returns time/pairs having loss of spacings.

    Parameters
    ----------
    sim_ids : pd.DataFrame
        timeseries containing flight_id, x and y columns.
    horiz_lim : float
        Maximum horizontal spacing allowed.
    vert_lim : float
        Maximum vertical spacing allowed.

    Returns
    -------
    pd.DataFrame
        timeseries containg flight_id, horiz_spacing and vert_spacing of flight
        having a loss of spacing.
    """
    df = pd.DataFrame()
    df = sim_ids.apply(
        lambda row: get_loss_of_spacing_series(row, horiz_lim, vert_lim),
        axis="columns",
        result_type="expand",
    )
    column_names = ["flight_id", "horiz_spacing", "vert_spacing"]
    if df.shape[1] == 4:
        column_names.append("NSE")
    df.columns = column_names
    df["n"] = df["flight_id"].apply(len)
    df = df.query("n>0")
    df = df.drop(columns="n")

    df = df.explode(column_names)
    if len(df) > 0:
        df["flight_id_1"], df["flight_id_2"] = zip(*df["flight_id"])
        if "NSE" in column_names:
            df["NSE_1"], df["NSE_2"] = zip(*df["NSE"])
            df = df.drop(columns="NSE")
    return df.drop(columns="flight_id")


def wrap_get_los_df(args):
    return get_loss_of_spacing_df(*args)


@lru_cache
def loss_of_spacing(
    traffic: traffic.core.Traffic,
    max_horizontal_spacing: float,
    max_vertical_spacing: float,
    projection: Union[pyproj.Proj, crs.Projection, None] = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Compute loss of spacing dataframe.

    A loss of spacing happens when a pair of aircraft are getting closer
    than max_horizontal_spacing and max_vertical_spacing at the same time.

    Parameters
    ----------
    traffic : traffic.core.Traffic
        traffic object containg flights.
    max_horizontal_spacing : float
        maximum allowed horizontal spacing
    max_vertical_spacing : float
        maximum allowed vertical spacing
    projection : Union[pyproj.Proj, cartes.crs.Projection, None], optional
        You need to provide a decentprojection able to approximate distances by
        Euclide formula. By default, EuroPP() projection is considered, but a
        non explicit argument will raise a warning., by default None
    max_workers : int, optional
        Number of cpu cores to use, by default 4

    Returns
    -------.
    pd.DataFrame
        timeseries containg flight_id, horiz_spacing and vert_spacing of flight
        having a loss of spacing.
    """

    if projection is None:
        logging.warning("Defaulting to projection EuroPP()")
        projection = crs.EuroPP()

    extra_cols2agregate = None
    if "NSE" in traffic.data.columns:
        extra_cols2agregate = ("NSE",)
    simult_ids = et.simult_ids_from_traffic(
        traffic,
        projection=projection,
        max_workers=max_workers,
        extra_cols2agregate=extra_cols2agregate,
    )

    simult_ids["n"] = simult_ids["flight_id"].apply(len)
    simult_ids = simult_ids.query("n > 1")
    if max_workers == 1:
        return get_loss_of_spacing_df(
            simult_ids, max_horizontal_spacing, max_vertical_spacing
        )

    n_chunks = int(max_workers * 1)
    l_chunks = int(len(simult_ids) // n_chunks + 1)
    chunks = [
        (
            simult_ids.iloc[l_chunks * i : l_chunks * (i + 1)],
            max_horizontal_spacing,
            max_vertical_spacing,
        )
        for i in range(n_chunks)
    ]
    with mp.Pool(max_workers) as pool:
        l_res = pool.starmap(
            get_loss_of_spacing_df,
            tqdm(chunks, total=n_chunks, desc="loss of spacing"),
        )
    los_df = pd.concat(l_res)

    return los_df


def keep_traffic_close_encounters(
    traffic: traffic.core.Traffic,
    close_encounters: pd.DataFrame,
    max_workers=4,
) -> traffic.core.Traffic:

    
    groups = list(close_encounters.groupby("flight_id_1"))
    dico_filtering = {g[0]: list(g[1].index) for g in groups}
    groups2 = list(close_encounters.groupby("flight_id_2"))
    for g in groups2:
        if g[0] in dico_filtering:
            dico_filtering[g[0]] += list(g[1].index)
        else:
            dico_filtering[g[0]] = list(g[1].index)

    global DICO_FILTERING
    DICO_FILTERING = dico_filtering

    # Chunking the traffic object
    lf = []
    for f in tqdm(traffic):
        lf.append(f)
    l_chunks = len(traffic) // max_workers + 1
    flight_chunks = [
        lf[l_chunks * k : l_chunks * (k + 1)] for k in tqdm(range(max_workers))
    ]

    # // of the code
    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(filter_traffic_wrap, flight_chunks)
    return sum(results)


def filter_traffic_wrap(list_flights):
    traffic = Traffic.from_flights(list_flights)
    return traffic.iterate_lazy().pipe(filter_traffic).eval(max_workers=1)


def filter_traffic(flight):
    try:
        ts = list(DICO_FILTERING[flight.flight_id])
        return flight.query(f"timestamp.isin({ts})")
    except KeyError:
        return None
