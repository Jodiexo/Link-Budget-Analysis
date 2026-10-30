"""Core link budget math
Units: power in dBw, gains in dBi, losses in dB, distance in metres, frequency in Hz"""

import math

C = 299_752_458  # speed of light in m/s
K_DB = -228.6  # Boltzmann's constant in dB

def wavelength_m(freq_hz):
    return C / freq_hz

def fspl_db(distance_m, freq_hz):
    """Free space path loss in dB"""
    return 20 * math.log10(4 * math.pi * distance_m / wavelength_m(freq_hz))

def eirp_dbw(tx_power_dbw,tx_gain_dbi, tx_losses_db=0.0):
    """Effective isotropic radiated power in dBW"""
    return 10 * math.log10(tx_power_dbw) + tx_gain_dbi - tx_losses_db

def link_budget(distance_m, freq_hz, tx_power_w, tx_gain_dbi, rx_gain_dbi, sys_noise_temp_k, data_rate_bps, required_ebn0_db, tx_losses_db=0.0, rx_losses_db=0.0, other_losses_db=0.0):
    """Compute a full link budget at one distance. Returns a dict of results."""

    eirp = eirp_dbw(tx_power_w, tx_gain_dbi, tx_losses_db)
    fspl = fspl_db(distance_m, freq_hz)
    prx = eirp - fspl - other_losses_db + rx_gain_dbi - rx_losses_db
    t_db = 10 * math.log10(sys_noise_temp_k)
    cn0 = prx - K_DB - t_db  #carrier to noise density ratio in dB-Hz
    ebn0 = cn0 - 10 * math.log10(data_rate_bps)  #energy per bit to noise density

    return {
        "EIRP (dBW)": eirp,
        "Free-space path loss (dB)": fspl,
        "Received power (dBW)": prx,
        "G/T (dB/K)": prx - t_db,
        "C/N0 (dB-Hz)": cn0,
        "Eb/N0 (dB)": ebn0,
        "Link margin (dB)": ebn0 - required_ebn0_db
    }

def max_range_m(freq_hz, tx_power_w, tx_gain_dbi, rx_gain_dbi,
                sys_noise_temp_k, data_rate_bps, required_ebn0_db,
                required_margin_db=3.0, tx_losses_db=0.0, rx_losses_db=0.0,
                other_losses_db=0.0):

    """Longest distance at which the link still meets the required Eb/N0 and margin."""
    eirp = eirp_dbw(tx_power_w, tx_gain_dbi, tx_losses_db)
    allowed_fspl = (eirp - other_losses_db + rx_gain_dbi - rx_losses_db
                    - K_DB - 10 * math.log10(sys_noise_temp_k)
                    - 10 * math.log10(data_rate_bps)
                    - required_ebn0_db - required_margin_db)

    return wavelength_m(freq_hz) / (4 * math.pi) * 10 ** (allowed_fspl / 20)

if __name__ == "__main__":
    #Sanity Check; S-Band crosslink, 5W, 10 dBi antennas, 1 Mbps BPSK
    params = dict(freq_hz= 2.2e9, tx_power_w=5.0, tx_gain_dbi=10.0, rx_gain_dbi=10.0,
                  sys_noise_temp_k=500.0, data_rate_bps=1e6, required_ebn0_db=9.6)
    d = max_range_m(**params, required_margin_db=3.0)
    print(f"Max range (3 dB margin): {d/1000:,.0f} km")
    for k, v in link_budget(distance_m=d, **params).items():
        print(f" {k:28s} {v:8.2f} ")
                     