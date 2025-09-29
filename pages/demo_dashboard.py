
import warnings, datetime as dt
warnings.filterwarnings("ignore")
# Unified "Calls & Outcomes" filter, fast UI, violin KDE, true-scale toggle
# Caching: Uses both Streamlit @st.cache_data and pybaseball.cache for optimal performance
# ---------------------------------------------------------------------------------------------
# Unified “Calls & Outcomes” filter, fast UI, violin KDE, true-scale toggle
# ---------------------------------------------------------------------------------------------
import warnings, datetime as dt
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sequential, qualitative
from requests.exceptions import HTTPError

from pybaseball import (
    statcast,               # used only for tiny fallback roster window
    statcast_batter,        # per-batter Statcast
    batting_stats,          # FanGraphs team board (primary roster source)
    playerid_reverse_lookup,
    chadwick_register,      # cross IDs (mlbam, fangraphs, bref, retro)
    cache,                  # pybaseball caching system
)

# Enable pybaseball caching for faster data retrieval
cache.enable()

# -------- Page ----------
st.set_page_config(page_title="Batter Analytics Dashboard", layout="wide", page_icon="📈")

def render_navbar():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        .navbar {
            position: sticky;
            top: 0;
            z-index: 999;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 1rem 2rem;
            border-radius: 0 0 1rem 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .navbar-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .navbar-brand {
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            text-decoration: none;
        }
        
        .navbar-links {
            display: flex;
            gap: 2rem;
        }
        
        .navbar-links a {
            color: rgba(255, 255, 255, 0.9);
            text-decoration: none;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }
        
        .navbar-links a:hover {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            transform: translateY(-1px);
        }
    </style>
    
    <div class="navbar">
        <div class="navbar-content">
            <div class="navbar-brand">⚾ Baseball Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_navbar()

# Navigation buttons
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    if st.button("Home", key="nav_home", use_container_width=True):
        st.switch_page("Home.py")

with nav_col2:
    if st.button("Dashboard", key="nav_dashboard", use_container_width=True, type="primary"):
        pass  # Already on dashboard page

with nav_col3:
    if st.button("Pitch Prediction", key="nav_pitch", use_container_width=True):
        st.switch_page("pages/pitch_prediction.py")

with nav_col4:
    if st.button("Deception Index", key="nav_deception", use_container_width=True):
        st.switch_page("pages/deception_index.py")

st.markdown("---")



st.title("Select Team ➜ Player")
st.caption("Pick a team and hitter to explore 14-zone hot/cold performance, pitch locations, and contact quality. "
           "Each plot has its own date range control. Filters below the plate apply to both Pitch Location and Contact Quality.")
# add some bold text
st.markdown("**Note:** This dashboard uses automatically updated data. It may take a few moments to load if it has been inactive for a while.")

# -------- Constants -----
IN_ZONES = list(range(1,10))
OZ_ZONES = [11,12,13,14]
PLATE_HALF = 0.83  # ft (rulebook width 17 in ≈ 1.4167 ft; we render the hitting area ±0.83 ft)
TEAM_CODES = ["TOR","NYY","BOS","TBR","BAL","CHW","CLE","DET","KCR","MIN",
              "HOU","SEA","OAK","LAA","TEX","ATL","NYM","PHI","MIA","WSN",
              "CHC","STL","PIT","MIL","CIN","LAD","SFG","SDP","ARI","COL"]

# Baseball geometry for true-to-scale dots (ball radius in feet)
BASEBALL_DIAMETER_IN = 2.90
BASEBALL_RADIUS_FT   = (BASEBALL_DIAMETER_IN / 12.0) / 2.0  # inches → feet → radius

# -------- Top controls ---
cA, cB, cC = st.columns([1,1,1])
with cA:
    season = st.number_input("Season", min_value=2015, max_value=dt.date.today().year,
                             value=dt.date.today().year, step=1)
with cB:
    team = st.selectbox("Team", TEAM_CODES, index=TEAM_CODES.index("TOR"))
with cC:
    fallback_days = st.number_input("Fallback roster window (days)", min_value=7, max_value=60, value=21,
                                    help="Used only if FanGraphs team endpoint errors")

# -------- Helpers -------
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
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# ------ Team hitters (FG primary, Statcast fallback) ------
@st.cache_data(show_spinner=False)
def get_team_hitters_list_fg(team_code: str, season: int) -> pd.DataFrame:
    fg = batting_stats(season, season, team=team_code)
    if fg is None or fg.empty:
        return pd.DataFrame(columns=["display","mlbam","PA"])
    if "PA" in fg.columns:
        fg = fg[fg["PA"].fillna(0) > 0].copy()
    reg = chadwick_register()[["key_mlbam","key_fangraphs","name_first","name_last"]]
    reg["key_fangraphs"] = pd.to_numeric(reg["key_fangraphs"], errors="coerce")
    lut = reg.dropna(subset=["key_fangraphs"])
    key = "playerid" if "playerid" in fg.columns else ("IDfg" if "IDfg" in fg.columns else None)
    hitters = fg.merge(lut, left_on=key, right_on="key_fangraphs", how="left") if key else fg
    hitters["mlbam"] = pd.to_numeric(hitters.get("key_mlbam"), errors="coerce").astype("Int64")
    display = hitters["Name"] if "Name" in hitters.columns else (
        hitters["name_first"].str.title() + " " + hitters["name_last"].str.title()
    )
    hitters["display"] = display
    hitters = hitters.dropna(subset=["mlbam"]).drop_duplicates(subset=["mlbam"])
    hitters = hitters.sort_values(["PA"] if "PA" in hitters.columns else [], ascending=False)
    return hitters[["display","mlbam","PA"]]

@st.cache_data(show_spinner=False)
def get_team_hitters_list_fallback_statcast(team_code: str, days: int) -> pd.DataFrame:
    end = dt.date.today(); start = end - dt.timedelta(days=days)
    df = statcast(start_dt=start.strftime("%Y-%m-%d"), end_dt=end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return pd.DataFrame(columns=["display","mlbam","PA"])
    is_team_batting = (
        ((df["home_team"] == team_code) & (df["inning_topbot"] == "Bot")) |
        ((df["away_team"] == team_code) & (df["inning_topbot"] == "Top"))
    )
    d = df.loc[is_team_batting, ["batter","game_pk","at_bat_number"]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["display","mlbam","PA"])
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
        return get_team_hitters_list_fallback_statcast(team_code, fallback_days)
    except Exception:
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

# ------ Load one player’s season (single fetch, cached) ------
@st.cache_data(show_spinner=False)
def load_batter_season(season: int, mlbam_id: int) -> pd.DataFrame:
    s0, s1 = in_season_bounds(season)
    return statcast_batter(s0.strftime("%Y-%m-%d"), s1.strftime("%Y-%m-%d"), mlbam_id)

with st.spinner("Loading this hitter’s full season (Statcast)…"):
    season_df = load_batter_season(season, mlbam_id)

if season_df is None or season_df.empty:
    st.error("No per-batter Statcast rows for this player/season.")
    st.stop()

# ---------------- 14-zone Hot/Cold -----------------------
def zone_summary(df: pd.DataFrame, min_pitches=5) -> pd.DataFrame:
    need = ["zone","description","woba_value","woba_denom","launch_speed","launch_angle"]
    d = df[[c for c in df.columns if c in need]].dropna(subset=["zone"]).copy()
    swings = {"swinging_strike","swinging_strike_blocked","foul","foul_tip","foul_bunt","hit_into_play"}
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

def plot_zone14_rects(zs: pd.DataFrame, metric: str, base_df: pd.DataFrame, title: str):
    if base_df["sz_bot"].notna().any() and base_df["sz_top"].notna().any():
        zbot = float(base_df["sz_bot"].median()); ztop = float(base_df["sz_top"].median())
    else:
        zbot, ztop = 1.5, 3.5
    vals = zs.set_index("zone")[metric].astype(float).reindex(IN_ZONES + OZ_ZONES)
    if np.isfinite(vals.values).any():
        vmin, vmax = float(np.nanmin(vals.values)), float(np.nanmax(vals.values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax: vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=[[vmin, vmax]], x=[-999,-998], y=[-999,-998],
                             colorscale=sequential.Inferno, showscale=True, colorbar=dict(title=metric)))
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
    fig.add_shape(type="rect", x0=-PLATE_HALF, x1=PLATE_HALF, y0=zbot, y1=ztop,
                  line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=title, xaxis_title="plate_x (ft)", yaxis_title="plate_z (ft)",
        xaxis=dict(range=[-2.5,2.5], zeroline=False),
        yaxis=dict(range=[0,5], zeroline=False),
        plot_bgcolor="white", height=700
    )
    return equal_axes(fig)

# =========================
# PART 1 (collapsible): Hot/Cold by Zone
# =========================
with st.expander("Zone Based Hitter Metrics", expanded=False):
    hm_d1, hm_d2 = date_controls(season, "hm")
    metric = st.selectbox("Metric", ["wOBAcon","swing%","whiff%","in-play%","ev_avg","la_avg"],
                          index=0, key="hm_metric")
    hm_df = filter_by_dates(season_df, hm_d1, hm_d2)
    zs = zone_summary(hm_df, min_pitches=5)
    if zs.empty:
        st.info("Not enough pitches per zone in this range.")
    else:
        left, mid, right = st.columns([1,2,1])
        with mid:
            st.plotly_chart(
                plot_zone14_rects(zs, metric, hm_df, f"{player_name} — {metric}"),
                use_container_width=True
            )

# ---------------- Shared utilities for Part 2 ----------------
def call_label(desc: str) -> str:
    if desc in (None, np.nan): return "Other (pitch)"
    if desc == "ball" or desc == "blocked_ball": return "Ball"
    if desc == "called_strike": return "Called Strike"
    if desc in {"swinging_strike","swinging_strike_blocked"}: return "Swinging Strike"
    if desc in {"foul","foul_tip","foul_bunt"}: return "Foul"
    if desc == "hit_into_play": return "Ball in Play"
    return "Other (pitch)"

def outcome_label(ev: str) -> str:
    if not isinstance(ev, str) or ev == "":         return "No Hit"
    if ev == "home_run": return "Home Run"
    if ev == "triple":   return "Triple"
    if ev == "double":   return "Double"
    if ev == "single":   return "Single"
    if ev in {"walk","intent_walk"}: return "Walk"
    if ev == "hit_by_pitch":          return "HBP"
    if ev in {"strikeout","strikeout_double_play"}: return "Strikeout"
    return "No Hit"

def scatter_plate(df: pd.DataFrame, title: str, use_true_scale: bool, max_shapes: int):
    if df.empty: return None
    zbot = float(df["sz_bot"].median()) if df["sz_bot"].notna().any() else 1.5
    ztop = float(df["sz_top"].median()) if df["sz_top"].notna().any() else 3.5

    disp = np.where(df["Outcome"].ne("No Hit"), df["Outcome"], df["Call"])
    df = df.assign(DisplayClass=disp)

    cats = sorted(pd.Series(disp).dropna().unique().tolist())
    palette = qualitative.D3 if len(cats) <= 10 else qualitative.Alphabet
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

    dist_col = "hit_distance_sc" if "hit_distance_sc" in df.columns else ("hit_distance" if "hit_distance" in df.columns else None)
    hover = []
    for _, r in df.iterrows():
        dist_txt = f"<br>Distance: {int(r[dist_col])} ft" if (dist_col and pd.notna(r[dist_col])) else ""
        bs  = f"{int(r['balls'])}-{int(r['strikes'])}" if pd.notna(r['balls']) and pd.notna(r['strikes']) else ""
        inn = f"{int(r['inning'])} {r['inning_topbot']}" if pd.notna(r['inning']) and pd.notna(r['inning_topbot']) else ""
        hover.append(
            f"Class: {r['DisplayClass']}<br>Pitch: {r.get('pitch_name','')}"
            f"<br>Desc: {r.get('description','')}<br>Event: {r.get('events','')}"
            f"<br>Count: {bs}<br>Inning: {inn}{dist_txt}"
        )

    if (not use_true_scale) or (len(df) > max_shapes):
        fig = px.scatter(df, x="plate_x", y="plate_z", color="DisplayClass",
                         opacity=0.8, color_discrete_map=color_map, render_mode="webgl")
        fig.update_traces(customdata=np.array(hover, dtype=object), hovertemplate="%{customdata}<extra></extra>")
    else:
        fig = go.Figure()
        for cat in cats:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(color=color_map[cat], size=10, opacity=0.9),
                                     name=str(cat), showlegend=True, hoverinfo="skip"))
        r = BASEBALL_RADIUS_FT
        for _, row in df.iterrows():
            cx = float(row["plate_x"]); cy = float(row["plate_z"])
            col = color_map.get(row["DisplayClass"], "#666")
            fig.add_shape(type="circle", xref="x", yref="y",
                          x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                          line=dict(color=col, width=1.0), fillcolor=col, opacity=0.75)
        fig.add_trace(go.Scatter(x=df["plate_x"], y=df["plate_z"], mode="markers",
                                 marker=dict(size=1, opacity=0), showlegend=False,
                                 customdata=np.array(hover, dtype=object),
                                 hovertemplate="%{customdata}<extra></extra>"))

    fig.add_shape(type="rect", xref="x", yref="y",
                  x0=-PLATE_HALF, x1=PLATE_HALF, y0=zbot, y1=ztop,
                  line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=title, plot_bgcolor="white",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
        height=700
    )
    fig.update_xaxes(range=[-2.5, 2.5], title="plate_x (ft)")
    fig.update_yaxes(range=[0, 5], title="plate_z (ft)", scaleanchor="x", scaleratio=1)
    return fig

@st.cache_data(show_spinner=False)
def _sample_for_kde(series_by_group: dict, max_per_group: int = 2000):
    out = {}
    rng = np.random.RandomState(42)
    for k, s in series_by_group.items():
        vals = pd.to_numeric(s, errors="coerce").dropna().values
        if len(vals) == 0: continue
        if len(vals) > max_per_group:
            idx = rng.choice(len(vals), size=max_per_group, replace=False)
            vals = vals[idx]
        out[k] = vals
    return out

def violin_kde(series_by_group: dict, title: str, xlabel: str, height=420):
    sampled = _sample_for_kde(series_by_group, max_per_group=2000)
    if not sampled: return None
    frames = [pd.DataFrame({xlabel: arr, "Group": k}) for k, arr in sampled.items()]
    d = pd.concat(frames, ignore_index=True)
    fig = px.violin(d, x="Group", y=xlabel, color="Group", box=False, points=False)
    fig.update_layout(title=title, height=height,
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"))
    return fig

def general_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pa = df.dropna(subset=["game_pk","at_bat_number"]).drop_duplicates(subset=["game_pk","at_bat_number"]).shape[0]
    total_pitches = len(df)
    swing_set = {"swinging_strike","swinging_strike_blocked","foul","foul_tip","foul_bunt","hit_into_play"}
    swings_n = df["description"].isin(swing_set).sum()
    inplay_n  = df["type"].eq("X").sum() if "type" in df.columns else np.nan
    events = df["events"].fillna("")
    counts = {
        "single": "Single", "double": "Double", "triple": "Triple", "home_run": "Home Run",
        "walk": "Walk", "intent_walk": "Intentional Walk", "hit_by_pitch": "HBP",
        "strikeout": "Strikeout", "strikeout_double_play": "Strikeout DP",
        "field_out": "Field Out", "force_out": "Force Out", "sac_fly": "Sac Fly",
        "sac_bunt": "Sac Bunt", "double_play": "Double Play",
        "grounded_into_double_play": "GIDP", "fielders_choice_out": "Fielder's Choice Out",
        "truncated_pa": "Truncated PA"
    }
    rows = [{"Metric":"Pitches","Value":total_pitches},
            {"Metric":"PA (approx)","Value":pa},
            {"Metric":"Swings","Value":swings_n},
            {"Metric":"Balls in play","Value":inplay_n}]
    for ev, label in counts.items():
        rows.append({"Metric":label, "Value":int((events == ev).sum())})
    return pd.DataFrame(rows)

# =========================
# PART 2 (collapsible): Pitch Location & Contact Quality (re-laid out)
# =========================
with st.expander("Pitch Location & Contact Quality", expanded=False):

    # Toggles for plate scatter
    st.markdown("### Pitch Location")
    speed_col1, speed_col2 = st.columns([1,1])
    with speed_col1:
        true_scale = st.toggle("True-scale balls (slower)", value=False,
                               help="Draw real-size balls with shapes. Off = fast WebGL markers.")
    with speed_col2:
        max_shapes = st.slider("Max true-scale balls", min_value=100, max_value=1200, value=400, step=100,
                               help="If selection exceeds this, auto-switch to WebGL markers.")

    sc_d1, sc_d2 = date_controls(season, "sc")
    sc_base = filter_by_dates(season_df, sc_d1, sc_d2)

    # Derived labels
    base_sc = sc_base.dropna(subset=["plate_x","plate_z"]).copy()
    base_sc["Call"]    = base_sc["description"].apply(call_label)
    base_sc["Outcome"] = base_sc["events"].apply(outcome_label)

    # -------- Unified Filters (shared with contact quality) --------
    st.markdown("### Filters (apply to Pitch Location & Contact Quality)")
    fc1, fc2, fc3 = st.columns([1,1,1])
    with fc1:
        balls_sel   = st.multiselect("Balls", [0,1,2,3], default=[0,1,2,3], key="balls")
    with fc2:
        strikes_sel = st.multiselect("Strikes", [0,1,2], default=[0,1,2], key="strikes")
    with fc3:
        two_strike = st.checkbox("Quick: Two-strike", value=False)
        if two_strike: strikes_sel = [2]

    fi1, fi2, fi3 = st.columns([1,1,1])
    with fi1:
        innings = sorted(base_sc["inning"].dropna().astype(int).unique().tolist() or list(range(1,10)))
        inning_sel = st.multiselect("Innings", innings, default=innings, key="inn")
    with fi2:
        topbot = st.radio("Half-inning", ["Both","Top","Bot"], horizontal=True, key="tb")
    with fi3:
        homeaway = st.radio("Venue", ["Both","Home","Away"], horizontal=True, key="ha")

    with st.expander("Pitch types (optional)"):
        all_types = sorted(base_sc["pitch_name"].dropna().unique().tolist())
        selected_types = st.multiselect("Show only these pitch types", all_types, default=all_types, key="sc_types")

    unified_options = [
        "Ball", "Called Strike", "Swinging Strike", "Foul", "Ball in Play", "Other (pitch)",
        "Single", "Double", "Triple", "Home Run", "Walk", "HBP", "Strikeout", "No Hit",
        "ground_ball", "fly_ball", "popup",
    ]
    selected_unified = st.multiselect("Include categories", unified_options, default=unified_options, key="unified")

    # ---- Apply filters in order ----
    sc = base_sc.copy()
    sc = sc[sc["balls"].isin(balls_sel) & sc["strikes"].isin(strikes_sel)]
    sc = sc[sc["inning"].astype("Int64").isin(pd.Series(inning_sel, dtype="Int64"))]
    if topbot != "Both":
        sc = sc[sc["inning_topbot"] == ("Top" if topbot == "Top" else "Bot")]
    is_home_pitch = sc["home_team"] == team
    venue_mask = (is_home_pitch) if homeaway == "Home" else (~is_home_pitch) if homeaway == "Away" else pd.Series(True, index=sc.index)
    sc = sc[venue_mask]
    if selected_types:
        sc = sc[sc["pitch_name"].isin(selected_types)]

    if selected_unified:
        sel = set(selected_unified)
        mask_calls = sc["Call"].isin(sel & {"Ball","Called Strike","Swinging Strike","Foul","Ball in Play","Other (pitch)"})
        mask_outcomes = sc["Outcome"].isin(sel & {"Single","Double","Triple","Home Run","Walk","HBP","Strikeout","No Hit"})
        bb_targets = sel & {"ground_ball","fly_ball","popup"}
        if bb_targets:
            mask_bb = sc["type"].eq("X") & sc["bb_type"].isin(bb_targets)
        else:
            mask_bb = pd.Series(False, index=sc.index)
        sc = sc[mask_calls | mask_outcomes | mask_bb]

    # ================= Layout change here =================
    # Left column: ALL Contact Quality plots stacked.
    # Right column: Strike-zone pitch scatter (plate).
    left_col, right_col = st.columns([1,1])

    # ---------- LEFT: Contact Quality ----------
    with left_col:
        st.markdown("## Contact Quality")
        st.caption("Violin (KDE) distributions for contact and swing speed. Reflects filters to the right.")

        # Contact-only slice for EV/LA
        contact_df = sc[sc["events"].isin(["single","double","triple","home_run"])].copy()
        contact_df = contact_df.assign(DisplayClass=np.where(contact_df["Outcome"].ne("No Hit"),
                                                             contact_df["Outcome"], contact_df["Call"]))

        # EV violin
        if not contact_df.empty and "launch_speed" in contact_df.columns:
            fig_ev = violin_kde({k: v["launch_speed"] for k, v in contact_df.groupby("DisplayClass")},
                                "Exit Velocity", "EV (mph)")
            if fig_ev is not None:
                st.plotly_chart(fig_ev, use_container_width=True)
            else:
                st.caption("Not enough EV data.")
        else:
            st.caption("No contact (EV) data in this selection.")

        # LA violin
        if not contact_df.empty and "launch_angle" in contact_df.columns:
            fig_la = violin_kde({k: v["launch_angle"] for k, v in contact_df.groupby("DisplayClass")},
                                "Launch Angle", "LA (°)")
            if fig_la is not None:
                st.plotly_chart(fig_la, use_container_width=True)
            else:
                st.caption("Not enough LA data.")
        else:
            st.caption("No contact (LA) data in this selection.")

        # Swing speed (swings only)
        st.markdown("### Swing Speed (KDE)")
        swing_mask = sc["description"].isin(
            ["swinging_strike","swinging_strike_blocked","foul","foul_tip","foul_bunt","hit_into_play"]
        )
        swings = sc.loc[swing_mask].copy().assign(
            DisplayClass=np.where(sc["Outcome"].ne("No Hit"), sc["Outcome"], sc["Call"])[swing_mask]
        )
        if "bat_speed" in swings.columns and swings["bat_speed"].notna().any():
            fig_bs = violin_kde({k: v["bat_speed"] for k, v in swings.groupby("DisplayClass")},
                                "Swing Speed", "Bat speed (mph)")
            if fig_bs is not None:
                st.plotly_chart(fig_bs, use_container_width=True)
            else:
                st.caption("Not enough swing speed data.")
        else:
            st.caption("Swing speed not available in this selection.")

    # ---------- RIGHT: Strike zone scatter ----------
    with right_col:
        fig_sc = scatter_plate(sc, f"{player_name} — pitch locations (filtered)", true_scale, max_shapes)
        if fig_sc is None:
            st.info("No plate_x/plate_z in this filtered range.")
        else:
            st.plotly_chart(fig_sc, use_container_width=True)

    # ---------- General stats (full width below) ----------
    st.markdown("## General Stats")
    st.dataframe(general_stats(sc), use_container_width=True)