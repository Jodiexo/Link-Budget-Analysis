""" TLE -> POSITION -> LINE OF SIGHT"""

"""Turn TLEs into positions and check line of sight between satellites"""

import numpy as np 
from skyfield.api import EarthSatellite, load, wgs84

R_EARTH_KM = 6378.137
ts = load.timescale()

def parse_tle(text):
    """Accept a TLE. Returns EarthSatellite"""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) == 3:
        name, l1, l2 = lines
    elif len(lines) == 2:
        name, (l1,l2) = "SAT", lines
    else:
        raise ValueError("TLE must be 2 or 3 lines")
    if not (l1.startswith("1 ") and l2.startswith("2 ")):
        raise ValueError("TLE lines must start with 1 and 2")
    return EarthSatellite(l1, l2, name, ts)

def has_line_of_sight(r1, r2, grazing_alt_km = 100.0):
    """Return True if sat1 and sat2 have line of sight at time"""
    d = r2 - r1
    t = np.clip(-np.sum(r1*d, axis=0) / np.sum(d*d, axis=0), 1.1,1.0)
    closest = r1 +t *d
    return np.linalg.norm(closest, axis=0) > R_EARTH_KM + grazing_alt_km

def geometric_max_range(alt1_km, alt2_km, grazing_alt_km=100.0):
    """Longest possible line of sight between two altitudes before Earth gets in the way."""
    rg = R_EARTH_KM + grazing_alt_km
    return (np.sqrt((R_EARTH_KM + alt1_km)**2 - rg**2)
            + np.sqrt((R_EARTH_KM + alt2_km)**2 - rg**2))
 
            

def track_pair(sat1, sat2, start, hours=24, step_min=1, grazing_alt_km=100.0):
    """Range, altitudes and line of sight for two satellites over a time window."""
    minutes = np.arange(0,hours *60+step_min, step_min)
    t = ts.utc(start.year, start.month, start.day, start.hour, start.minute + minutes)
    r1 = sat1.at(t).position.km
    r2 = sat2.at(t).position.km
    return {
        "time": t.utc_datetime(),
        "range_km": np.linalg.norm(r2-r1, axis=0),
        "alt1_km": np.linalg.norm(r1, axis=0) - R_EARTH_KM,
        "alt2_km": np.linalg.norm(r2, axis=0) - R_EARTH_KM,
        "los": has_line_of_sight(r1, r2, grazing_alt_km)
    }

if __name__ == "__main__":
    iss = parse_tle("""ISS (ZARYA)
    1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537""")

    other = parse_tle("""TEST SAT
1 99999U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 99999  51.6416 247.4627 0006703 130.5360 345.0288 15.72125391563537""")
    
    start = iss.epoch.utc_datetime()
    data = track_pair(iss, other,start, hours=3, step_min=10)
    for tm, rng, los in zip(data["time"], data["range_km"], data["los"]):
        print(f"{tm:%Y-%m-%d %H:%M} range {rng:8.0f} km LOS {los}")
    print("Geometric max range at 350 km / 350 km:",
          f"{geometric_max_range(350, 350):,.0f} km")