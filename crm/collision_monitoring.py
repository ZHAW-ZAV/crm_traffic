import multiprocessing as mp
import os
from math import cosh, exp, sinh

import numpy as np
from scipy.stats import laplace, ncx2, norm


def proba_h_olap_norm_norm_nse_circle(
    delta: float, NSE1: float, NSE2: float, ap_radius: float
) -> float:
    """Compute proba of horiz olap using normal distrib and cirle shape.

    Compute the probability for 2 airplanes separated by a distance delta and
    normally distributed NSE of respectively NSE1 and NSE2 to have a vertical
    overlap. The model assumes that airplanes are reprsented by circles of
    similar radius ap_radius.

    Parameters
    ----------
    delta : float
        Reported distance between the 2 airplanes in meters.
    NSE1:
        Navigation System Error for airplane 1.
    NSE2:
        Navigation System Error for airplane 2.
    ap_radius:
        Airplane 1 and 2 radius.

    Returns
    -------
    float
        Probability of horizontal overlap

    Notes
    -----
    Units should be similar for every parameters.
    """
    sigma1 = NSE1 / 1.96
    sigma2 = NSE2 / 1.96
    s2 = sigma1 ** 2 + sigma2 ** 2
    k = 2
    nc = delta ** 2 / s2
    dlim_conv = (ap_radius ** 2) / s2
    return ncx2.cdf(dlim_conv, df=k, nc=nc)


def proba_normal_normal_circle_mc(n_samp, distance, NSE1, NSE2, lambda_xy):
    np.random.seed((os.getpid()))

    n_samp = int(n_samp)
    mu_x1, mu_y1 = 0, 0
    mu_x2, mu_y2 = distance, 0

    variance_x1 = NSE1 / 1.96
    variance_y1 = NSE1 / 1.96
    variance_x2 = NSE2 / 1.96
    variance_y2 = NSE2 / 1.96

    x1, y1 = norm.rvs(mu_x1, variance_x1, n_samp), norm.rvs(
        mu_y1, variance_y1, n_samp
    )
    x2, y2 = norm.rvs(mu_x2, variance_x2, n_samp), norm.rvs(
        mu_y2, variance_y2, n_samp
    )

    dists_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
    p = np.sum(dists_squared < lambda_xy ** 2) / n_samp
    return p


def proba_normal_normal_circle_mc_para(
    n_samp, distance, NSE1, NSE2, lambda_xy
):
    num_processes = 50
    n_samp = int(n_samp)
    chunks = [
        (n_samp // num_processes, distance, NSE1, NSE2, lambda_xy)
        for _ in range(num_processes)
    ]
    with mp.Pool(num_processes) as pool:
        scores = pool.starmap(proba_normal_normal_circle_mc, chunks)
    return np.mean(scores)


def proba_h_olap_norm_norm_nse_square(
    delta: float, NSE1: float, NSE2: float, lambda_xy: float
) -> float:
    """Compute proba of horiz olap using normal distrib and square shape.

    Compute the probability for 2 airplanes separated by a distance delta and
    normally distributed NSE of respectively NSE1 and NSE2 to have a vertical
    overlap. The model assumes that airplanes are reprsented by square of
    length lambda_xy.

    Parameters
    ----------
    delta : float
        Reported distance between the 2 airplanes in meters.
    NSE1:
        Navigation System Error for airplane 1.
    NSE2:
        Navigation System Error for airplane 2.
    lambda_xy:
        Airplane 1 and 2 length.

    Returns
    -------
    float
        Probability of horizontal overlap

    Notes
    -----
    Units should be similar for every parameters.
    """
    sigma1 = NSE1 / 1.96
    sigma2 = NSE2 / 1.96
    sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2)
    olap_y = norm.cdf(+lambda_xy, 0, sigma) - norm.cdf(-lambda_xy, 0, sigma)
    olap_x = norm.cdf(lambda_xy, delta, sigma) - norm.cdf(
        -lambda_xy, delta, sigma
    )
    return olap_x * olap_y


def proba_normal_normal_square_mc(n_samp, distance, NSE1, NSE2, lambda_xy):
    np.random.seed((os.getpid()))
    n_samp = int(n_samp)
    mu_x1, mu_y1 = 0, 0
    mu_x2, mu_y2 = distance, 0

    variance_x1 = NSE1 / 1.96
    variance_y1 = NSE1 / 1.96
    variance_x2 = NSE2 / 1.96
    variance_y2 = NSE2 / 1.96

    x1, y1 = norm.rvs(mu_x1, variance_x1, n_samp), norm.rvs(
        mu_y1, variance_y1, n_samp
    )
    x2, y2 = norm.rvs(mu_x2, variance_x2, n_samp), norm.rvs(
        mu_y2, variance_y2, n_samp
    )

    p = (
        np.sum((np.abs(x1 - x2) < lambda_xy) & (np.abs(y1 - y2) < lambda_xy))
        / n_samp
    )
    return p


def proba_normal_normal_square_mc_para(
    n_samp, distance, NSE1, NSE2, lambda_xy
):
    num_processes = 50
    n_samp = int(n_samp)
    chunks = [
        (n_samp // num_processes, distance, NSE1, NSE2, lambda_xy)
        for _ in range(num_processes)
    ]
    with mp.Pool(num_processes) as pool:
        scores = pool.starmap(proba_normal_normal_square_mc, chunks)
    return np.mean(scores)


def overlap_laplace_laplace(d, w, lam1, lam2):
    """Overlap laplace with laplace distribution.

    The overlap probabilities for
    a laplace distribution with a laplace distribution separated a distance d.

        * SASP-WGH-17-2010-May-Montreal-IP-01
        author: Steve Barry

    Parameters
    ----------
    d: float
        d between mean positions
    w : float
        approximate width of the aircraft
    lam1: float
        scale parameter for the first laplace distribution
    lam2: float
        scale parameter for the second laplace distribution

    Returns
    -------
    y : float

    """
    prob = g3(d, w, lam1, lam2)
    return prob


def g3(d, w, lam1, lam2):
    """Compute term used in laplace laplace overlap computation.

    A term in the overlap probabilities for
    a Laplace distribution with a Laplace distribution separated a distance d.

        * SASP-WGH-17-2010-May-Montreal-IP-01
        author: Steve Barry

    Parameters
    ----------
    d: float
        d between mean positions
    w : float
        approximate width of the aircraft
    lam1: float
        scale parameter for the first laplace distribution
    lam2: float
        scale parameter for the second laplace distribution

    Returns
    -------
    y : float

    """
    if np.abs(lam1 - lam2) < 1e-3:
        return g4(d, w, lam1)
    if d >= w:
        y1 = exp(-d / lam2) * lam2 ** 2 * sinh(w / lam2)
        y1 -= exp(-d / lam1) * lam1 ** 2 * sinh(w / lam1)
        y1 /= lam2 * lam2 - lam1 * lam1
        y = y1
    else:
        y1 = exp(-w / lam1) * lam1 ** 2 * cosh(d / lam1)
        y1 -= exp(-w / lam2) * lam2 ** 2 * cosh(d / lam2)
        y1 /= lam2 * lam2 - lam1 * lam1
        y = 1.0 + y1
    return y


def g4(d, w, lam):
    """Compute term used in laplace laplace overlap computation.

    This term is a component in the overlap probabilities for
    a laplace distribution with a laplace distribution separated a distance d.

        * SASP-WGH-17-2010-May-Montreal-IP-01
        author: Steve Barry

    Parameters
    ----------
    d: float
        d between mean positions
    w : float
        approximate width of the aircraft
    lam: float
        scale parameter for the laplace distribution

    Returns
    -------
    y : float

    """
    if d >= w:
        y1 = exp(-d / lam) / 2.0 / lam
        y2 = (d + 2 * lam) * sinh(w / lam) - w * cosh(w / lam)
        y = y1 * y2
    else:
        temp1 = d * sinh(d / lam) - (w + 2.0 * lam) * cosh(d / lam)
        y = 1.0 + exp(-w / lam) / 2.0 / lam * temp1
    return y


def proba_h_olap_laplace_laplace_nse_square(
    distance: float, NSE1: float, NSE2: float, lambda_xy: float
) -> float:
    """Compute proba of horiz olap using laplace distrib and square shape.

    Compute the probability for 2 airplanes separated by a distance delta and
    laplace distributed NSE of respectively NSE1 and NSE2 to have a vertical
    overlap. The model assumes that airplanes are reprsented by square of
    length lambda_xy.

    Parameters
    ----------
    delta : float
        Reported distance between the 2 airplanes in meters.
    NSE1:
        Navigation System Error for airplane 1.
    NSE2:
        Navigation System Error for airplane 2.
    lambda_xy:
        Airplane 1 and 2 length.

    Returns
    -------
    float
        Probability of horizontal overlap

    Notes
    -----
    Units should be similar for every parameters.
    """
    lam1 = NSE1 / np.log(20)
    lam2 = NSE2 / np.log(20)
    p_olap_x = overlap_laplace_laplace(distance, lambda_xy, lam1, lam2)
    p_olap_y = overlap_laplace_laplace(0, lambda_xy, lam1, lam2)
    return p_olap_x * p_olap_y


def proba_laplace_laplace_square_mc(n_samp, distance, NSE1, NSE2, lambda_xy):
    np.random.seed((os.getpid()))
    n_samp = int(n_samp)
    mu_x1, mu_y1 = 0, 0
    mu_x2, mu_y2 = distance, 0

    variance_x1 = NSE1 / np.log(20)
    variance_y1 = NSE1 / np.log(20)
    variance_x2 = NSE2 / np.log(20)
    variance_y2 = NSE2 / np.log(20)

    x1, y1 = laplace.rvs(mu_x1, variance_x1, n_samp), laplace.rvs(
        mu_y1, variance_y1, n_samp
    )
    x2, y2 = laplace.rvs(mu_x2, variance_x2, n_samp), laplace.rvs(
        mu_y2, variance_y2, n_samp
    )
    cond = (np.abs(x1 - x2) < lambda_xy) & (np.abs(y1 - y2) < lambda_xy)
    p = np.sum(cond) / n_samp
    return p


def proba_laplace_laplace_square_mc_para(
    n_samp, distance, NSE1, NSE2, lambda_xy
):
    """Apply a function in // to a dict of traj objects."""
    num_processes = 50
    n_samp = int(n_samp)
    chunks = [
        (n_samp // num_processes, distance, NSE1, NSE2, lambda_xy)
        for _ in range(num_processes)
    ]
    with mp.Pool(num_processes) as pool:
        scores = pool.starmap(proba_laplace_laplace_square_mc, chunks)
    return np.mean(scores)


def nacp2epu(nacp: int) -> float:
    """Compute EPU fron NACp.

    Parameters
    ----------
    nacp : int
        NACp.

    Returns
    -------
    float
        Radius of containment in NM.

    """
    if nacp == 1:
        epu = 10
    elif nacp == 2:
        epu = 4
    elif nacp == 3:
        epu = 2
    elif nacp == 4:
        epu = 1
    elif nacp == 5:
        epu = 0.5
    elif nacp == 6:
        epu = 0.3
    elif nacp == 7:
        epu = 0.1
    elif nacp == 8:
        epu = 0.05
    elif nacp == 9:
        epu = 30 / 1852
    elif nacp == 10:
        epu = 10 / 1852
    elif nacp == 11:
        epu = 3 / 1852
    else:
        return np.nan
    return epu


def radius_error_95_norm(n_samp, NSE1):
    n_samp = int(n_samp)
    mu_x1, mu_y1 = 0, 0

    variance_x1 = NSE1 / 1.96
    variance_y1 = NSE1 / 1.96

    x1, y1 = norm.rvs(mu_x1, variance_x1, n_samp), norm.rvs(
        mu_y1, variance_y1, n_samp
    )

    return sorted(np.sqrt(x1 ** 2 + y1 ** 2))[int(n_samp * 0.95)]


def epu2nse_norm(epu: float) -> float:
    """Convert EPU value into a NSE value for normal distribution.

    Converts an Estimated Position Uncertainty value into a Navigation System
    Error. The EPU represents a radius of error while NSE is defined
    component-wise.

    Parameters
    ----------
    epu : float
        EPU value.

    Returns
    -------
    float
        NSE value.
    """
    return epu / 1.2489


def radius_error_95_laplace(n_samp, NSE1):
    n_samp = int(n_samp)
    mu_x1, mu_y1 = 0, 0

    variance_x1 = NSE1 / np.log(20)
    variance_y1 = NSE1 / np.log(20)

    x1, y1 = laplace.rvs(mu_x1, variance_x1, n_samp), laplace.rvs(
        mu_y1, variance_y1, n_samp
    )

    return sorted(np.sqrt(x1 ** 2 + y1 ** 2))[int(n_samp * 0.95)]


def epu2nse_laplace(epu: float) -> float:
    """Convert EPU value into a NSE value for laplace distribution.

    Converts an Estimated Position Uncertainty value into a Navigation System
    Error. The EPU represents a radius of error while NSE is defined
    component-wise.

    Parameters
    ----------
    epu : float
        EPU value.

    Returns
    -------
    float
        NSE value.
    """
    return epu / 1.2984


def proba_vert_olap_norm(
    vert_spacing: float, sigma_oe: float, lambda_z: float
) -> float:
    """Compute the probability of vertical overlap

    This function returns the probability of having a vertical overlap for 2
    airplanes of height lambda_z separated by an observed vert_spacing and
    having an observed error following a normal distribution of mean 0 and
    standard deviation sigma_oe.

    Parameters
    ----------
    vert_spacing : float
        Difference of observed altitude between the 2 aircraft
    sigma_oe : float
        Observed vertical error standard deviation
    lambda_z : float
        Height of aircraft

    Returns
    -------
    float
        Probability of vertical overlap
    """
    lz2 = lambda_z ** 2
    s2 = 2 * sigma_oe ** 2
    lz2 = lambda_z ** 2
    nc = vert_spacing ** 2 / s2
    return ncx2.cdf(lz2 / s2, df=1, nc=nc)


def proba_vert_olap_laplace_sum(vert_spacing: float) -> float:
    """Compute the probability of vertical overlap

    This function returns the probability of having a vertical overlap for 2
    airplanes of height lambda_z=30ft separated by an observed vert_spacing.
    This overlap models considers a Correspondance Error and an Altimetry
    System Error following laplace distrbutions with means of 0 and laplace
    scaling of 12.68ft and 28.28ft respectively.

    The model has been obtained using monte-carlo simulation and uses a
    polynomial approximation.

    Parameters
    ----------
    vert_spacing : float
        Difference of observed altitude between the 2 aircraft in ft.


    Returns
    -------
    float
        Probability of vertical overlap
    """
    poly_log = np.array(
        [
            -3.97815541e-14,
            1.75594199e-11,
            3.81173767e-08,
            -3.46117464e-05,
            -4.12930734e-03,
            -3.26108142e-01,
        ]
    )

    return 10 ** np.poly1d(poly_log)(vert_spacing)


def proba_vert_olap_normal_mc(n_samp, vert_spacing, sigma_oe, lambda_z):
    np.random.seed((os.getpid()))

    n_samp = int(n_samp)
    mu_z1 = 0
    mu_z2 = vert_spacing

    variance_z1 = sigma_oe
    variance_z2 = sigma_oe

    z1 = norm.rvs(mu_z1, variance_z1, n_samp)
    z2 = norm.rvs(mu_z2, variance_z2, n_samp)

    p = np.sum(np.abs(z1 - z2) < lambda_z) / n_samp
    return p


def proba_vert_olap_normal_mc_para(n_samp, vert_spacing, sigma_oe, lambda_z):
    num_processes = 50
    n_samp = int(n_samp)
    chunks = [
        (n_samp // num_processes, vert_spacing, sigma_oe, lambda_z)
        for _ in range(num_processes)
    ]
    with mp.Pool(num_processes) as pool:
        scores = pool.starmap(proba_vert_olap_normal_mc, chunks)
    return np.mean(scores)


def proba_vert_olap_laplace_mc(
    n_samp, vert_spacing, lambda_CR_lap, lambda_ASE_lap, lambda_z
):
    np.random.seed((os.getpid()))

    n_samp = int(n_samp)

    z1 = laplace.rvs(loc=0, scale=lambda_CR_lap, size=n_samp) + laplace.rvs(
        loc=0, scale=lambda_ASE_lap, size=n_samp
    )
    z2 = laplace.rvs(loc=0, scale=lambda_CR_lap, size=n_samp) + laplace.rvs(
        loc=0, scale=lambda_ASE_lap, size=n_samp
    )
    z2 = z2 + vert_spacing
    p = np.sum(np.abs(z1 - z2) < lambda_z) / n_samp
    return p


def proba_vert_olap_laplace_mc_para(
    n_samp, vert_spacing, lambda_CR_lap, lambda_ASE_lap, lambda_z
):
    num_processes = 50
    n_samp = int(n_samp)
    chunks = [
        (
            n_samp // num_processes,
            vert_spacing,
            lambda_CR_lap,
            lambda_ASE_lap,
            lambda_z,
        )
        for _ in range(num_processes)
    ]
    with mp.Pool(num_processes) as pool:
        scores = pool.starmap(proba_vert_olap_laplace_mc, chunks)
    return np.mean(scores)
