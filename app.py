# Streamlit — Hitter Hot/Cold & Pitch Location (team dropdown, FG fallback, single-player load)
# ---------------------------------------------------------------------------------------------

import warnings, datetime as dt, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sequential
from plotly.figure_factory import create_distplot
from requests.exceptions import HTTPError

from pybaseball import (
    statcast,               # used ONLY for tiny fallback roster window
    statcast_batter,        # per-batter Statcast
    batting_stats,          # FanGraphs team board (primary roster source)
    playerid_reverse_lookup,
    chadwick_register,      # cross-IDs (mlbam, fangraphs, bref, retro)
)

st.set_page_config(page_title="Hitter Hot/Cold — Team → Player", layout="wide")
st.title("Player Batting Analysis Dashboard")
st.caption("Pick a team and hitter to explore 14-zone hot/cold performance and pitch-location patterns. "
           "Use the per-chart date range and filters to focus on recent form or specific scenarios.")

# ----------------------- Constants -----------------------
IN_ZONES = list(range(1,10))
OZ_ZONES = [11,12,13,14]
PLATE_HALF = 0.83
TEAM_CODES = ["TOR","NYY","BOS","TBR","BAL","CHW","CLE","DET","KCR","MIN",
              "HOU","SEA","OAK","LAA","TEX","ATL","NYM","PHI","MIA","WSN",
              "CHC","STL","PIT","MIL","CIN","LAD","SFG","SDP","ARI","COL"]

# ----------------------- Top controls --------------------
st.title("Hitter Hot/Cold — Team → Player")

cA, cB, cC = st.columns([1,1,1])
with cA:
    season = st.number_input("Season", min_value=2015, max_value=dt.date.today().year,
                             value=dt.date.today().year, step=1)
with cB:
    team = st.selectbox("Team", TEAM_CODES, index=TEAM_CODES.index("TOR"))
with cC:
    fallback_days = st.number_input("Fallback roster window (days)", min_value=7, max_value=60, value=21,
                                    help="Used only if FanGraphs team endpoint errors")

# ----------------------- Helpers -------------------------
def in_season_bounds(year: int):
    return dt.date(year, 3, 1), dt.date(year, 11, 30)

def date_controls(season: int, key_prefix: str):
    s0, s1 = in_season_bounds(season)
    st.caption("Date range for this chart")
    quick = st.segmented_control("Quick range", ["Season","Last 30","Last 14","This Month","Custom"],
                                 default="Season", key=f"{key_prefix}_q")
    if quick == "Season":
        return s0, s1
    if quick == "Last 30":
        end = min(dt.date.today(), s1); return max(s0, end - dt.timedelta(days=30)), end
    if quick == "Last 14":
        end = min(dt.date.today(), s1); return max(s0, end - dt.timedelta(days=14)), end
    if quick == "This Month":
        today = min(dt.date.today(), s1); start = dt.date(today.year, today.month, 1)
        return max(s0, start), today
    d1, d2 = st.date_input("Custom", (s0, s1), min_value=s0, max_value=s1, key=f"{key_prefix}_dates")
    if isinstance(d1, tuple): d1, d2 = d1[0], d1[1]
    return d1, d2

def filter_by_dates(df: pd.DataFrame, d1: dt.date, d2: dt.date) -> pd.DataFrame:
    if df is None or df.empty: return df
    g = pd.to_datetime(df["game_date"]).dt.date
    return df.loc[(g >= d1) & (g <= d2)].copy()

def equal_axes(fig: go.Figure) -> go.Figure:
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # keep plate undistorted
    return fig

# ---------------- Team hitters dropdown (FG primary, Statcast fallback) ----------------
@st.cache_data(show_spinner=False)
def get_team_hitters_list_fg(team_code: str, season: int) -> pd.DataFrame:
    # FanGraphs team board (fast when available)
    fg = batting_stats(season, season, team=team_code)
    if fg is None or fg.empty:
        return pd.DataFrame(columns=["display","mlbam","PA"])
    if "PA" in fg.columns:
        fg = fg[fg["PA"].fillna(0) > 0].copy()  # avoid pitchers-only
    # Map FG -> MLBAM using Chadwick (best-in-class crosswalk)
    reg = chadwick_register()[["key_mlbam","key_fangraphs","name_first","name_last"]]
    reg["key_fangraphs"] = pd.to_numeric(reg["key_fangraphs"], errors="coerce")
    lut = reg.dropna(subset=["key_fangraphs"])
    if "playerid" in fg.columns:
        hitters = fg.merge(lut, left_on="playerid", right_on="key_fangraphs", how="left")
    else:
        hitters = fg.merge(lut, left_on="IDfg", right_on="key_fangraphs", how="left") if "IDfg" in fg.columns else fg
    hitters["mlbam"] = pd.to_numeric(hitters.get("key_mlbam"), errors="coerce").astype("Int64")
    display = hitters["Name"] if "Name" in hitters.columns else (
        hitters["name_first"].str.title() + " " + hitters["name_last"].str.title()
    )
    hitters["display"] = display
    hitters = hitters.dropna(subset=["mlbam"]).drop_duplicates(subset=["mlbam"])
    sort_cols = ["PA"] if "PA" in hitters.columns else []
    hitters = hitters.sort_values(sort_cols, ascending=False)
    return hitters[["display","mlbam","PA"]]

@st.cache_data(show_spinner=False)
def get_team_hitters_list_fallback_statcast(team_code: str, days: int) -> pd.DataFrame:
    # Minimal Statcast (recent window) to find who actually batted for the team.
    end = dt.date.today(); start = end - dt.timedelta(days=days)
    df = statcast(start_dt=start.strftime("%Y-%m-%d"), end_dt=end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return pd.DataFrame(columns=["display","mlbam","PA"])
    # keep only when team is batting
    is_team_batting = (
        ((df["home_team"] == team_code) & (df["inning_topbot"] == "Bot")) |
        ((df["away_team"] == team_code) & (df["inning_topbot"] == "Top"))
    )
    d = df.loc[is_team_batting, ["batter","game_pk","at_bat_number"]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["display","mlbam","PA"])
    # estimate PA via unique (game, ab#)
    pa = d.groupby(["game_pk","batter","at_bat_number"]).size().reset_index()[["batter","at_bat_number"]]
    pa = pa.groupby("batter").size().reset_index(name="PA")
    ids = pa["batter"].astype(int).tolist()
    lut = playerid_reverse_lookup(ids, key_type="mlbam")[["key_mlbam","name_first","name_last"]]
    lut["display"] = lut["name_first"].str.title() + " " + lut["name_last"].str.title()
    out = pa.merge(lut, left_on="batter", right_on="key_mlbam", how="left").rename(columns={"batter":"mlbam"})
    out = out[["display","mlbam","PA"]].drop_duplicates().sort_values("PA", ascending=False)
    return out

@st.cache_data(show_spinner=False)
def get_team_hitters_list(team_code: str, season: int, fallback_days: int) -> pd.DataFrame:
    try:
        return get_team_hitters_list_fg(team_code, season)
    except HTTPError:
        # FanGraphs 500 or similar — fall back seamlessly
        return get_team_hitters_list_fallback_statcast(team_code, fallback_days)
    except Exception:
        # Any unexpected parse/network error: also fall back
        return get_team_hitters_list_fallback_statcast(team_code, fallback_days)

hitters_df = get_team_hitters_list(team, season, fallback_days)
if hitters_df.empty:
    st.error("Couldn’t build the team hitters list (FG error and fallback empty). Try another team or expand fallback window.")
    st.stop()

sel_idx = st.selectbox(
    "Player",
    range(len(hitters_df)),
    index=0,
    format_func=lambda i: f"{hitters_df.iloc[i]['display']} — MLBAM {int(hitters_df.iloc[i]['mlbam'])} — PA {int(hitters_df.iloc[i]['PA']) if pd.notna(hitters_df.iloc[i]['PA']) else 0}"
)
mlbam_id = int(hitters_df.iloc[sel_idx]["mlbam"])
player_name = hitters_df.iloc[sel_idx]["display"]

# ---------------- Single-player season Statcast (one load) ----------------
@st.cache_data(show_spinner=False)
def load_batter_season(season: int, mlbam_id: int) -> pd.DataFrame:
    s0, s1 = in_season_bounds(season)
    return statcast_batter(s0.strftime("%Y-%m-%d"), s1.strftime("%Y-%m-%d"), mlbam_id)

with st.spinner("Loading this hitter’s full season (Statcast)…"):
    season_df = load_batter_season(season, mlbam_id)

if season_df is None or season_df.empty:
    st.error("No per-batter Statcast rows for this player/season.")
    st.stop()

# ---------------- Zone summary helper ----------------
def zone_summary(df: pd.DataFrame, min_pitches=5) -> pd.DataFrame:
    need = ["zone","description","woba_value","woba_denom","launch_speed","launch_angle"]
    d = df[[c for c in df.columns if c in need]].dropna(subset=["zone"]).copy()
    swings = {"swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play"}
    whiffs = {"swinging_strike","swinging_strike_blocked"}
    d["swing"]  = d["description"].isin(swings)
    d["whiff"]  = d["description"].isin(whiffs)
    d["inplay"] = d["description"].eq("hit_into_play")
    g = (d.groupby("zone", as_index=False)
           .agg(n=("zone","size"),
                swings=("swing","sum"),
                whiffs=("whiff","sum"),
                inplay=("inplay","sum"),
                woba_num=("woba_value","sum"),
                woba_den=("woba_denom","sum"),
                ev_avg=("launch_speed","mean"),
                la_avg=("launch_angle","mean")))
    g = g[g["n"] >= min_pitches]
    g["swing%"]   = g["swings"]/g["n"]
    g["whiff%"]   = g["whiffs"]/g["swings"].replace(0, np.nan)
    g["in-play%"] = g["inplay"]/g["n"]
    g["wOBAcon"]  = np.where(g["woba_den"]>0, g["woba_num"]/g["woba_den"], np.nan)
    return g[g["zone"].isin(IN_ZONES + OZ_ZONES)].copy()

# ---------------- Hot/Cold (14 zones, centered, taller) ----------------
st.markdown("## Hot/Cold by Zone")
hm_d1, hm_d2 = date_controls(season, "hm")
metric = st.selectbox("Metric", ["wOBAcon","swing%","whiff%","in-play%","ev_avg","la_avg"], index=0, key="hm_metric")
hm_df = filter_by_dates(season_df, hm_d1, hm_d2)
zs = zone_summary(hm_df, min_pitches=5)

def plot_zone14_rects(zs: pd.DataFrame, metric: str, base_df: pd.DataFrame, title: str):
    # strike zone height from medians
    if base_df["sz_bot"].notna().any() and base_df["sz_top"].notna().any():
        zbot = float(base_df["sz_bot"].median()); ztop = float(base_df["sz_top"].median())
    else:
        zbot, ztop = 1.5, 3.5

    vals = zs.set_index("zone")[metric].astype(float).reindex(IN_ZONES + OZ_ZONES)
    if np.isfinite(vals.values).any():
        vmin, vmax = float(np.nanmin(vals.values)), float(np.nanmax(vals.values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    fig = go.Figure()
    # colorbar stub
    fig.add_trace(go.Heatmap(z=[[vmin, vmax]], x=[-999,-998], y=[-999,-998],
                             colorscale=sequential.Inferno, showscale=True, colorbar=dict(title=metric)))

    # inner 3x3 tiles
    x_edges = np.linspace(-PLATE_HALF, PLATE_HALF, 4)
    y_edges = np.linspace(zbot, ztop, 4)
    order = {1:(0,2),2:(1,2),3:(2,2),4:(0,1),5:(1,1),6:(2,1),7:(0,0),8:(1,0),9:(2,0)}

    def val2color(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "rgba(0,0,0,0.05)"
        t = (v - vmin) / (vmax - vmin) if vmax>vmin else 0.5
        t = min(max(t, 0.05), 0.95)
        return sequential.Inferno[int(t*(len(sequential.Inferno)-1))]

    for z,(i,j) in order.items():
        v = float(vals.loc[z]) if z in vals.index and np.isfinite(vals.loc[z]) else np.nan
        fig.add_shape(type="rect",
                      x0=x_edges[i], x1=x_edges[i+1], y0=y_edges[j], y1=y_edges[j+1],
                      line=dict(color="white", width=0.6), fillcolor=val2color(v))

    # bands 11 above, 12 below, 13 left, 14 right
    bands = {
        11: (-PLATE_HALF,  PLATE_HALF,  ztop,           min(ztop+0.8, 5.0)),
        12: (-PLATE_HALF,  PLATE_HALF,  max(zbot-0.8,0), zbot),
        13: (max(-2.5, -PLATE_HALF-0.8), -PLATE_HALF,   zbot, ztop),
        14: ( PLATE_HALF,  min( 2.5,  PLATE_HALF+0.8),  zbot, ztop),
    }
    for z,(x0,x1,y0,y1) in bands.items():
        v = float(vals.loc[z]) if z in vals.index and np.isfinite(vals.loc[z]) else np.nan
        fig.add_shape(type="rect",
                      x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color="white", width=0.6), fillcolor=val2color(v))

    # rulebook strike zone
    fig.add_shape(type="rect", x0=-PLATE_HALF, x1=PLATE_HALF, y0=zbot, y1=ztop,
                  line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")

    fig.update_layout(
        title=title,
        xaxis_title="plate_x (ft)", yaxis_title="plate_z (ft)",
        xaxis=dict(range=[-2.5,2.5], zeroline=False),
        yaxis=dict(range=[0,5], zeroline=False),
        plot_bgcolor="white",
        height=700  # ~40% taller
    )
    return equal_axes(fig)

if zs.empty:
    st.info("Not enough pitches per zone in this range.")
else:
    # center at ~50% width
    left, mid, right = st.columns([1,2,1])
    with mid:
        st.plotly_chart(plot_zone14_rects(zs, metric, hm_df, f"{player_name} — {metric}"), use_container_width=True)

# ---------------- Pitch Location (mutually exclusive + granular filters) ---------------
st.markdown("## Pitch Location")
sc_d1, sc_d2 = date_controls(season, "sc")
sc_base = filter_by_dates(season_df, sc_d1, sc_d2)

# Build categorical labels on the UNFILTERED base (prevents empty-legend crashes)
def label_call(desc: str) -> str:
    if desc == "ball": return "Ball"
    if desc == "called_strike": return "Called Strike"
    if desc in {"swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play"}:
        return "Swinging"
    return "Other"

def label_outcome(ev: str) -> str:
    if ev == "home_run": return "Home Run"
    if ev in {"double","triple"}: return "Extra Base Hit"
    if ev == "single": return "Single"
    if ev in {"walk","hit_by_pitch"}: return "BB/HBP"
    if pd.isna(ev) or ev == "": return "No Hit"
    return "In-Play Other"

base_sc = sc_base.dropna(subset=["plate_x","plate_z"]).copy()
base_sc["CallClass"] = base_sc["description"].apply(label_call)
base_sc["Outcome"]   = base_sc["events"].apply(label_outcome)

# Choose ONE grouping (mutually exclusive)
mode = st.radio("Group points by", ["Calls","Outcomes"], horizontal=True, key="sc_mode")

# Pitch type filter (default all)
with st.expander("Pitch types (optional)"):
    all_types = sorted(base_sc["pitch_name"].dropna().unique().tolist())
    selected_types = st.multiselect("Show only these pitch types", all_types, default=all_types, key="sc_types")

# Explicit category checkboxes (built from base to avoid empty legends)
group_col = "CallClass" if mode == "Calls" else "Outcome"
all_groups = sorted(base_sc[group_col].dropna().unique().tolist())
chosen_groups = st.multiselect("Show only these categories", all_groups, default=all_groups, key="sc_groups")
if not chosen_groups:
    chosen_groups = all_groups[:]  # keep non-empty to avoid removing all traces

# Apply filters
sc = base_sc.copy()
if selected_types:
    sc = sc[sc["pitch_name"].isin(selected_types)]
sc = sc[sc[group_col].isin(chosen_groups)]

def scatter_plate(df: pd.DataFrame, group_col: str, title: str):
    if df.empty: return None
    zbot = float(df["sz_bot"].median()) if df["sz_bot"].notna().any() else 1.5
    ztop = float(df["sz_top"].median()) if df["sz_top"].notna().any() else 3.5
    fig = px.scatter(df, x="plate_x", y="plate_z", color=group_col, opacity=0.78,
                     labels={"plate_x":"plate_x (ft)","plate_z":"plate_z (ft)"})
    # strike zone
    fig.add_shape(type="rect", x0=-PLATE_HALF, x1=PLATE_HALF, y0=zbot, y1=ztop,
                  line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")

    # Legend pinned to the RIGHT (x/xanchor/y/yanchor per Plotly legend API)
    fig.update_layout(
        title=title, plot_bgcolor="white",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
        height=700  # **match Hot/Cold height**
    )
    # Equal x/y scale (no plate distortion)
    fig.update_xaxes(range=[-2.5,2.5]); fig.update_yaxes(range=[0,5], scaleanchor="x", scaleratio=1)
    return fig

# Center the plot in the middle column (same as Hot/Cold layout)
left_sc, mid_sc, right_sc = st.columns([1,2,1])
with mid_sc:
    fig_sc = scatter_plate(sc, group_col, f"{player_name} — pitch locations ({mode.lower()})")
    if fig_sc is None:
        st.info("No plate_x/plate_z in this filtered range.")
    else:
        st.plotly_chart(fig_sc, use_container_width=True)

# ---------------- Contact quality (KDE; robust against short data) ----------------
# ---------------- Contact quality (KDE; side-by-side) ----------------
st.markdown("## Contact Quality")
st.caption("Smoothed (KDE) distributions of exit velocity and launch angle for selected outcomes "
           "or swing classes in the current date window and filters.")

swing_mask = sc["description"].isin(["swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play"])
contact = sc[sc["events"].isin(["single","double","triple","home_run"])].copy()

from plotly.figure_factory import create_distplot  # KDE curves (Plotly FF)

def kde_overlay(series_by_group: dict[str, pd.Series], title: str, xlabel: str):
    data, labels = [], []
    for lbl, ser in series_by_group.items():
        vals = pd.to_numeric(ser, errors="coerce").dropna().values
        if len(vals) >= 5:  # a few points for stable KDE
            data.append(vals); labels.append(lbl)
    if not data:
        return None
    fig = create_distplot(data, labels, show_hist=False, show_rug=False, curve_type="kde")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Density",
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
                      height=420)
    return fig

# Show the two curves side-by-side using Streamlit columns
lc, rc = st.columns([1,1])
if mode == "Outcomes":
    if not contact.empty:
        with lc:
            fig_ev = kde_overlay({k: v["launch_speed"] for k, v in contact.groupby("Outcome")}, "Exit Velocity", "EV (mph)")
            if fig_ev: st.plotly_chart(fig_ev, use_container_width=True)
            else: st.caption("Not enough data for EV KDE.")
        with rc:
            fig_la = kde_overlay({k: v["launch_angle"] for k, v in contact.groupby("Outcome")}, "Launch Angle", "LA (°)")
            if fig_la: st.plotly_chart(fig_la, use_container_width=True)
            else: st.caption("Not enough data for LA KDE.")
    else:
        st.caption("No contact events in this selection.")
else:
    swings = sc.loc[swing_mask]
    if "bat_speed" in swings.columns and swings["bat_speed"].notna().any():
        with lc:
            fig_bs = kde_overlay({k: v["bat_speed"] for k, v in swings.groupby("CallClass")}, "Swing Speed", "Bat speed (mph)")
            if fig_bs: st.plotly_chart(fig_bs, use_container_width=True)
            else: st.caption("Not enough swing speed data.")
        with rc:
            st.caption("Select Outcomes to view EV/LA KDE.")
    else:
        st.caption("Swing speed not available for this selection.")
