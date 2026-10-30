"""Satellite-to-satellite link budget app.

Run with:  streamlit run app.py
"""
from datetime import datetime, time, timezone

import math
import altair as alt
import pandas as pd
import streamlit as st

from linkbudget import link_budget, max_range_m
from orbit import parse_tle, track_pair, geometric_max_range

SERIES = "#2a78d6"      # range / margin line
CONTACT = "#1baf7a"     # shading for contact windows
LIMIT = "#52514e"       # dashed limit lines

SAMPLE_TLE_1 = """ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""
SAMPLE_TLE_2 = """TEST SAT
1 99999U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 99999  51.6416 287.4627 0006703 130.5360 325.0288 15.72125391563537"""

# Typical Eb/N0 needed for a bit error rate of 1e-5 (uncoded)
MODULATIONS = {"BPSK / QPSK": 9.6, "8PSK": 13.0, "16QAM": 13.4, "Custom": None}

st.set_page_config(page_title="Sat-to-Sat Link Budget", layout="wide")
st.title("Satellite-to-satellite link budget")

# ------------------------------------------------------------------ inputs
# ------------------------------------------------------------------ inputs
# Tooltip text for the help icon next to each input. Rule of thumb used below:
# +6 dB doubles the range, +3 dB stretches it by about 40 %.
HELP = {
    "freq": "Carrier frequency of the link.\n\n"
            "Typical bands: UHF ~0.4 GHz, S ~2–4, X ~8–12, Ku ~12–18, Ka ~26–40.\n\n"
            "**Effect:** with the same antenna gains, doubling the frequency adds 6 dB of "
            "path loss and halves the range. (A dish of fixed size gains more at higher "
            "frequency, which can offset this — update the antenna gains to match.)",
    "power": "RF power out of satellite 1's transmitter, in watts.\n\n"
             "Typical: CubeSats 1–5 W, larger satellites 10–200 W.\n\n"
             "**Effect:** doubling power adds 3 dB → about 40 % more range.",
    "tx_gain": "How strongly satellite 1's antenna focuses energy toward satellite 2, "
               "in dB relative to an antenna radiating equally in all directions (dBi).\n\n"
               "Typical: patch/omni 0–6 dBi, helix or patch array 10–15 dBi, "
               "1 m dish at X-band ~36 dBi.\n\n"
               "**Effect:** +6 dB doubles the range — but a narrower beam must be pointed "
               "more accurately.",
    "rx_gain": "How much signal satellite 2's antenna collects from satellite 1's "
               "direction, in dBi. Same scale as transmit gain.\n\n"
               "**Effect:** +6 dB doubles the range.",
    "t_sys": "Total noise of satellite 2's receiver, expressed as a temperature in kelvin. "
             "Includes the electronics plus whatever the antenna sees (deep space is cold; "
             "Earth in the beam adds ~290 K).\n\n"
             "Typical: 300–1000 K for spacecraft receivers.\n\n"
             "**Effect:** lower is better. Halving it adds 3 dB → about 40 % more range.",
    "rate": "How fast data is sent, in megabits per second.\n\n"
            "**Effect:** each bit gets less energy at higher rates. 10× the data rate "
            "costs 10 dB → range drops to about a third. Slowing down is the cheapest "
            "way to reach farther.",
    "mod": "How bits are put onto the carrier. Each choice sets the **Required Eb/N0** — "
           "the signal quality needed for about 1 error per 100,000 bits (no error "
           "correction).\n\n"
           "**BPSK/QPSK** (9.6 dB): most robust, used on most crosslinks. "
           "**8PSK / 16QAM**: more bits per symbol, but need a stronger signal.\n\n"
           "Pick **Custom** to enter your own value, e.g. if you use error-correcting codes.",
    "ebn0": "Energy per bit divided by noise density — the signal quality the receiver "
            "needs to decode reliably.\n\n"
            "Uncoded BPSK/QPSK ≈ 9.6 dB. With modern error-correcting codes (e.g. LDPC, "
            "turbo) it can be as low as 1–4 dB.\n\n"
            "**Effect:** every 6 dB lower doubles the range.",
    "margin": "Safety buffer on top of the required Eb/N0 to cover things the model "
              "doesn't capture (aging hardware, pointing jitter, interference).\n\n"
              "Typical: 3 dB; 6 dB for a conservative design.\n\n"
              "**Effect:** the link only counts as 'up' when it has at least this much "
              "spare. More margin = shorter usable range.",
    "tx_loss": "Signal lost in cables, connectors and filters between satellite 1's "
               "transmitter and its antenna.\n\nTypical: 0.5–2 dB.",
    "rx_loss": "Signal lost in cables, connectors and filters between satellite 2's "
               "antenna and its receiver.\n\nTypical: 0.5–2 dB.",
    "other": "Catch-all for real-world losses: antennas not pointed perfectly at each "
             "other, polarization mismatch, and imperfect hardware.\n\nTypical: 1–3 dB.",
}

with st.sidebar:
    st.header("Radio")
    st.caption("Hover the help icon next to any input for an explanation. "
               "Rule of thumb: **+6 dB doubles the range.**")
    freq_ghz = st.number_input("Frequency (GHz)", 0.1, 100.0, 2.2, 0.1, help=HELP["freq"])
    tx_power_w = st.number_input("Transmit power (W)", 0.01, 10000.0, 5.0, help=HELP["power"])
    tx_gain = st.number_input("Transmit antenna gain (dBi)", -10.0, 70.0, 20.0,
                              help=HELP["tx_gain"])
    rx_gain = st.number_input("Receive antenna gain (dBi)", -10.0, 70.0, 20.0,
                              help=HELP["rx_gain"])
    t_sys = st.number_input("Receiver system noise temp (K)", 10.0, 5000.0, 500.0,
                            help=HELP["t_sys"])
    rate_mbps = st.number_input("Data rate (Mbps)", 0.0001, 10000.0, 1.0, format="%.4f",
                                help=HELP["rate"])
    mod = st.selectbox("Modulation", list(MODULATIONS), help=HELP["mod"])
    req_ebn0 = (st.number_input("Required Eb/N0 (dB)", -5.0, 30.0, 9.6, help=HELP["ebn0"])
                if MODULATIONS[mod] is None else MODULATIONS[mod])
    if MODULATIONS[mod] is not None:
        st.caption(f"Required Eb/N0: {req_ebn0} dB")
    req_margin = st.number_input("Required link margin (dB)", 0.0, 20.0, 3.0,
                                 help=HELP["margin"])

    with st.expander("Losses"):
        tx_loss = st.number_input("Transmit line loss (dB)", 0.0, 20.0, 1.0,
                                  help=HELP["tx_loss"])
        rx_loss = st.number_input("Receive line loss (dB)", 0.0, 20.0, 1.0,
                                  help=HELP["rx_loss"])
        other_loss = st.number_input("Pointing / polarization / other (dB)", 0.0, 30.0, 1.0,
                                     help=HELP["other"])

    st.header("Geometry")
    grazing_km = st.number_input("Grazing altitude (km)", 0.0, 1000.0, 100.0,
                                 help="The line between the satellites must stay this far above "
                                      "Earth's surface. 100 km keeps it out of the atmosphere.")

col1, col2 = st.columns(2)
tle1_text = col1.text_area("Satellite 1 TLE (transmitter)", SAMPLE_TLE_1, height=110)
tle2_text = col2.text_area("Satellite 2 TLE (receiver)", SAMPLE_TLE_2, height=110)

c1, c2, c3, c4 = st.columns(4)
start_mode = c1.radio("Start time", ["TLE 1 epoch", "Now", "Custom (UTC)"])
custom_date = c2.date_input("Date (UTC)", disabled=start_mode != "Custom (UTC)")
custom_time = c2.time_input("Time (UTC)", time(0, 0), disabled=start_mode != "Custom (UTC)")
hours = c3.number_input("Duration (hours)", 1, 168, 24)
step_min = c4.number_input("Step (minutes)", 1, 60, 1)

# ------------------------------------------------------------------ parse
try:
    sat1, sat2 = parse_tle(tle1_text), parse_tle(tle2_text)
except Exception as e:
    st.error(f"Could not read a TLE: {e}")
    st.stop()

if start_mode == "TLE 1 epoch":
    start = sat1.epoch.utc_datetime().replace(second=0, microsecond=0)
elif start_mode == "Now":
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
else:
    start = datetime.combine(custom_date, custom_time, tzinfo=timezone.utc)

for sat in (sat1, sat2):
    age_days = abs((start - sat.epoch.utc_datetime()).total_seconds()) / 86400
    if age_days > 7:
        st.warning(f"{sat.name}: start time is {age_days:,.0f} days from this TLE's epoch. "
                   "SGP4 accuracy drops quickly after a few days — use a fresher TLE.")

radio = dict(freq_hz=freq_ghz * 1e9, tx_power_w=tx_power_w, tx_gain_dbi=tx_gain,
             rx_gain_dbi=rx_gain, sys_noise_temp_k=t_sys, data_rate_bps=rate_mbps * 1e6,
             required_ebn0_db=req_ebn0, tx_losses_db=tx_loss, rx_losses_db=rx_loss,
             other_losses_db=other_loss)

# ------------------------------------------------------------------ compute
data = track_pair(sat1, sat2, start, hours=hours, step_min=step_min, grazing_alt_km=grazing_km)
df = pd.DataFrame(data)
df = df.dropna(subset=["range_km"])  # SGP4 returns NaN if a TLE has decayed
if df.empty:
    st.error("SGP4 could not propagate these TLEs at this time. Check the TLEs and start time.")
    st.stop()

power_km = max_range_m(required_margin_db=req_margin, **radio) / 1000
geo_km = float(geometric_max_range(df.alt1_km.mean(), df.alt2_km.mean(), grazing_km))
max_km = min(power_km, geo_km)

df["margin_db"] = [link_budget(distance_m=r * 1000, **radio)["Link margin (dB)"]
                   for r in df.range_km]
df["link_up"] = df.los & (df.margin_db >= req_margin)
df["time"] = pd.to_datetime(df.time)

# ------------------------------------------------------------------ headline numbers
st.subheader("Maximum communication range")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Max usable range", f"{max_km:,.0f} km",
          help="The shorter of the power limit and the line-of-sight limit.")
m2.metric("Power limit", f"{power_km:,.0f} km",
          help=f"Distance where Eb/N0 falls to {req_ebn0} dB + {req_margin} dB margin.")
m3.metric("Line-of-sight limit", f"{geo_km:,.0f} km",
          help="Farthest two satellites at these altitudes can be before Earth blocks them.")
m4.metric("Time link is up", f"{df.link_up.mean() * 100:.0f} %")
st.caption("Limiting factor: " + ("**transmit power / antennas** — more power, gain or a lower "
           "data rate would extend range." if power_km < geo_km else
           "**Earth blocking the line of sight** — more power won't help; higher orbits would."))

# ------------------------------------------------------------------ contact windows
runs = (df.link_up != df.link_up.shift()).cumsum()
windows = []
for _, g in df[df.link_up].groupby(runs[df.link_up]):
    windows.append({"Start (UTC)": g.time.iloc[0], "End (UTC)": g.time.iloc[-1],
                    "Duration (min)": len(g) * step_min,
                    "Closest (km)": round(g.range_km.min()),
                    "Best margin (dB)": round(g.margin_db.max(), 1)})
win_df = pd.DataFrame(windows)

# ------------------------------------------------------------------ charts
shade = (alt.Chart(win_df).mark_rect(color=CONTACT, opacity=0.15)
         .encode(x="Start (UTC):T", x2="End (UTC):T")) if not win_df.empty else None


def limit_rule(value):
    return (alt.Chart(pd.DataFrame({"y": [value]}))
            .mark_rule(color=LIMIT, strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q"))


hover = alt.selection_point(fields=["time"], nearest=True, on="pointerover", empty=False)

range_line = alt.Chart(df).mark_line(color=SERIES, strokeWidth=2).encode(
    x=alt.X("time:T", title="Time (UTC)"),
    y=alt.Y("range_km:Q", title="Range between satellites (km)"))
range_pts = alt.Chart(df).mark_point(color=SERIES, size=60, filled=True).encode(
    x="time:T", y="range_km:Q",
    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    tooltip=[alt.Tooltip("time:T", title="UTC", format="%Y-%m-%d %H:%M"),
             alt.Tooltip("range_km:Q", title="Range (km)", format=",.0f"),
             alt.Tooltip("margin_db:Q", title="Margin (dB)", format=".1f"),
             alt.Tooltip("los:N", title="Line of sight"),
             alt.Tooltip("link_up:N", title="Link up")]).add_params(hover)
layers = [range_line, range_pts, limit_rule(max_km)]
if shade is not None:
    layers.insert(0, shade)
st.subheader("Range over time")
st.caption(f"Green bands = link up (line of sight and enough margin). "
           f"Dashed line = max usable range ({max_km:,.0f} km).")
st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")

margin_df = df.assign(margin_db=df.margin_db.where(df.los))  # hide margin when Earth blocks
margin_chart = alt.Chart(margin_df).mark_line(color=SERIES, strokeWidth=2).encode(
    x=alt.X("time:T", title="Time (UTC)"),
    y=alt.Y("margin_db:Q", title="Link margin (dB)"),
    tooltip=[alt.Tooltip("time:T", title="UTC", format="%Y-%m-%d %H:%M"),
             alt.Tooltip("margin_db:Q", title="Margin (dB)", format=".1f")])
st.subheader("Link margin over time")
st.caption(f"Gaps = Earth is blocking the line of sight. "
           f"Dashed line = required margin ({req_margin} dB).")
st.altair_chart(alt.layer(margin_chart, limit_rule(req_margin))
                .properties(height=260), width="stretch")

# ------------------------------------------------------------------ tables
st.subheader("Contact windows")
if win_df.empty:
    st.info("No contact windows in this time span.")
else:
    fmt = "%Y-%m-%d %H:%M"
    st.dataframe(win_df.assign(**{c: win_df[c].dt.strftime(fmt)
                                  for c in ("Start (UTC)", "End (UTC)")}),
                 hide_index=True, width="stretch")

st.subheader("Link budget at the max usable range",
             help="Follows the signal from satellite 1's transmitter to satellite 2's decoder, "
                  "at the max usable range. Read it top to bottom: each line builds on the ones "
                  "above. Hover a row to see how it's calculated with your current inputs.")
lb = link_budget(distance_m=max_km * 1000, **radio)
p_dbw = 10 * math.log10(tx_power_w)
t_db = 10 * math.log10(t_sys)
r_db = 10 * math.log10(rate_mbps * 1e6)

LB_HELP = {
    "EIRP (dBW)":
        "**Effective isotropic radiated power** — how strong the transmitted signal is in "
        "satellite 2's direction, counting both power and antenna focusing.\n\n"
        f"= transmit power + transmit gain − transmit line loss  \n"
        f"= {p_dbw:.2f} dBW + {tx_gain:.2f} dBi − {tx_loss:.2f} dB "
        f"= **{lb['EIRP (dBW)']:.2f} dBW**\n\n"
        f"({tx_power_w:g} W = {p_dbw:.2f} dBW)",
    "Free-space path loss (dB)":
        "**How much the signal weakens by spreading out** over the distance between the "
        "satellites. Usually the biggest number in the budget.\n\n"
        f"= 20·log10(4π × distance ÷ wavelength), at {max_km:,.0f} km and {freq_ghz:g} GHz "
        f"= **{lb['Free-space path loss (dB)']:.2f} dB**\n\n"
        "Doubling the distance or the frequency adds 6 dB.",
    "Received power (dBW)":
        "**Signal power arriving at satellite 2's receiver.** Very small — a negative dBW "
        "value like −120 dBW is about a millionth of a millionth of a watt.\n\n"
        "= EIRP − path loss − other losses + receive gain − receive line loss  \n"
        f"= {lb['EIRP (dBW)']:.2f} − {lb['Free-space path loss (dB)']:.2f} − {other_loss:.2f} "
        f"+ {rx_gain:.2f} − {rx_loss:.2f} = **{lb['Received power (dBW)']:.2f} dBW**",
    "G/T (dB/K)":
        "**Receiver figure of merit** — antenna gain divided by noise temperature. Higher is "
        "better: a big antenna and a quiet receiver.\n\n"
        f"= receive gain − 10·log10(noise temp)  \n"
        f"= {rx_gain:.2f} − 10·log10({t_sys:g} K) = {rx_gain:.2f} − {t_db:.2f} "
        f"= **{lb['G/T (dB/K)']:.2f} dB/K**",
    "C/N0 (dB-Hz)":
        "**Carrier-to-noise density** — how far the signal stands above the receiver's "
        "noise, per hertz of bandwidth.\n\n"
        "= received power − Boltzmann's constant − 10·log10(noise temp)  \n"
        f"= {lb['Received power (dBW)']:.2f} − (−228.6) − {t_db:.2f} "
        f"= **{lb['C/N0 (dB-Hz)']:.2f} dB-Hz**",
    "Eb/N0 (dB)":
        "**Energy per bit to noise density** — signal quality for each bit. This is what "
        "decides whether bits decode correctly.\n\n"
        "= C/N0 − 10·log10(data rate)  \n"
        f"= {lb['C/N0 (dB-Hz)']:.2f} − 10·log10({rate_mbps * 1e6:,.0f} bps) "
        f"= {lb['C/N0 (dB-Hz)']:.2f} − {r_db:.2f} = **{lb['Eb/N0 (dB)']:.2f} dB**",
    "Link margin (dB)":
        "**Spare signal quality** beyond what the modulation needs. Positive = the link "
        "closes; negative = too many bit errors.\n\n"
        "= Eb/N0 − required Eb/N0  \n"
        f"= {lb['Eb/N0 (dB)']:.2f} − {req_ebn0:.2f} = **{lb['Link margin (dB)']:.2f} dB**\n\n"
        f"Your required margin is {req_margin:g} dB. If the range here is set by power, this "
        "equals it exactly; if it's set by line of sight, there's extra to spare.",
}

with st.container(border=True):
    for item, value in lb.items():
        name_col, value_col = st.columns([3, 1])
        name_col.markdown(item, help=LB_HELP[item])
        value_col.markdown(f"<div style='text-align:right'>{value:,.2f}</div>",
                           unsafe_allow_html=True)

with st.expander("Raw time series"):
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), "link_timeseries.csv")