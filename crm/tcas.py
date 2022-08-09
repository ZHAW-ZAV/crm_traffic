from dataclasses import dataclass

import numpy as np


@dataclass
class TCAS_limits:
    """Contains tcas limits set.

    Attributes
    ----------
    tau_ta : float
        Tau value for Traffic Advisory [s].
    dmod_ta : float
        DMOD value for Traffic Advisory [NM].
    zthr_ta : float
        ZTHR value for Traffic Advisory [ft].
    tau_ra : float
        Tau value for Resolution Advisory [s].
    dmod_ra : float
        DMOD value for Resolution Advisory [NM].
    zthr_ra : float
        ZTHR value for Resolution Advisory [ft].
    tvthr_ra : np.array
        Reduced Time to Co-altitude Threshold for TA
    tvthr_ra : np.array
        Reduced Time to Co-altitude Threshold for RA
    """

    tau_ta: float
    dmod_ta: float
    zthr_ta: float
    tau_ra: float
    dmod_ra: float
    zthr_ra: float
    tvthr_ta: float
    tvthr_ra: float


def get_tcas_limits(alt: float, roc: float, alt_agl: float) -> TCAS_limits:
    """Return TCAS limits depending on altitude.

    Parameters
    ----------
    alt : float
        Aircraft altitude in [ft]
    roc : float
        Aircraft rate of climb in [ft/m]
    alt_agl : float
        Aircraft altitude above ground level in [ft]

    Returns
    -------
    TCAS_limits
        TCAS_limits limits instance.

    Private function to get TA/RA limits according to
    https://www.eurocontrol.int/publication/airborne-collision-avoidance-system-acas-guide
    """
    # When alt agl is low it is the used one to determine tcas limits
    if alt_agl < 2350:
        alt = alt_agl

    if alt < 1000:
        tau_ta, dmod_ta, zthr_ta = 20, 0.30, 850
        tau_ra, dmod_ra, zthr_ra = 0, 0., 0  # No RA at this level
        tvthr_ra = 0  # No RA at this level
    elif alt < 2350:
        tau_ta, dmod_ta, zthr_ta = 25, 0.33, 850
        tau_ra, dmod_ra, zthr_ra = 15, 0.20, 600
        tvthr_ra = 15
    elif alt < 5000:
        tau_ta, dmod_ta, zthr_ta = 30, 0.48, 850
        tau_ra, dmod_ra, zthr_ra = 20, 0.35, 600
        tvthr_ra = 18
    elif alt < 10000:
        tau_ta, dmod_ta, zthr_ta = 40, 0.75, 850
        tau_ra, dmod_ra, zthr_ra = 25, 0.55, 600
        tvthr_ra = 20
    elif alt < 20000:
        tau_ta, dmod_ta, zthr_ta = 45, 1.00, 850
        tau_ra, dmod_ra, zthr_ra = 30, 0.80, 600
        tvthr_ra = 22
    elif alt < 42000:
        tau_ta, dmod_ta, zthr_ta = 48, 1.30, 850
        tau_ra, dmod_ra, zthr_ra = 35, 1.10, 700
        tvthr_ra = 25
    else:
        tau_ta, dmod_ta, zthr_ta = 48, 1.30, 1200
        tau_ra, dmod_ra, zthr_ra = 35, 1.10, 800
        tvthr_ra = 25

    # The tvthr is only used for RA and when the roc is < 6ft/min
    if np.abs(roc) > 6:  #
        tvthr_ra = tau_ra
    tvthr_ta = tau_ta  # The tvthr is only used fo RA
    return TCAS_limits(
        tau_ta, dmod_ta, zthr_ta, tau_ra, dmod_ra, zthr_ra, tvthr_ta, tvthr_ra
    )
