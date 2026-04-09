"""
Hyrox Season 8 (2025-2026) – All Events – Interactive Dashboard
Run: python3 hyrox_dashboard.py  → ouvre http://127.0.0.1:8050
"""

import os, io, base64, unicodedata
from functools import lru_cache
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

def normalize_str(s):
    """Normalise une chaîne : minuscules, sans accents."""
    return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower()

# ── Chargement des données ─────────────────────────────────────────────────────
DATA_PATH = "hyrox_data.parquet"
pd.options.mode.chained_assignment = None

WORKOUT_COLS   = ["SkiErg_sec","SledPush_sec","SledPull_sec","BurpeeBJ_sec",
                  "Row_sec","FarmersCarry_sec","SandbagLunges_sec","WallBalls_sec"]
WORKOUT_LABELS = ["Ski Erg","Sled Push","Sled Pull","Burpee BJ",
                  "Row","Farmers Carry","Sandbag Lunges","Wall Balls"]
RUN_COLS    = [f"Run{i}_sec" for i in range(1, 9)]
RUN_LABELS  = [f"Run {i}" for i in range(1, 9)]
SCORE_COLS  = ["SkiErg_Score","SledPush_Score","SledPull_Score","BurpeeBJ_Score",
               "Row_Score","FarmersCarry_Score","SandbagLunges_Score","WallBalls_Score"]

print("Chargement des données...", flush=True)
_raw = pd.read_parquet(DATA_PATH)
# Convertir les colonnes category → str pour éviter les erreurs fillna sur Categorical
for _c in ["Country", "Age_Group", "Category", "Event", "Name", "Team_Name", "Finish_Time"]:
    if _c in _raw.columns:
        _raw[_c] = _raw[_c].astype(str).replace("nan", "")
df = _raw.reset_index(drop=True)
del _raw

# Pré-calculs au démarrage
df["Total_min"]    = df["Total_sec"] / 60
df["Runs_min"]     = df["Runs_Total_sec"] / 60
df["Workouts_min"] = df["Workouts_Total_sec"] / 60
df["Roxzone_min"]  = df["Roxzone_sec"] / 60
df["Display_Name"] = df.apply(
    lambda r: r["Team_Name"] if r.get("Team_Name", "") not in ("", "nan") else r["Name"], axis=1
)
df["Country"]   = df["Country"].replace("", "").fillna("")
df["Age_Group"] = df["Age_Group"].replace("", "–").fillna("–")
ALL_INDICES = df.index.tolist()
print(f"OK : {len(df)} lignes", flush=True)

# ── Ordre chronologique des événements ────────────────────────────────────────
EVENT_DATES = {
    "Hamburg 2025":    "2025-09-06",
    "Paris 2025":      "2025-09-20",
    "London 2025":     "2025-10-04",
    "Toronto 2025":    "2025-10-18",
    "Frankfurt 2025":  "2025-11-01",
    "Birmingham 2025": "2025-11-15",
    "Boston 2025":     "2025-11-22",
    "Dallas 2025":     "2025-12-06",
    "Nice 2026":       "2026-01-17",
    "Bologna 2026":    "2026-02-07",
    "Miami 2026":      "2026-02-21",
    "Houston 2026":    "2026-03-07",
    "Cape Town 2026":  "2026-03-21",
}
EVENTS_CHRONO = [e for e in sorted(EVENT_DATES, key=EVENT_DATES.get)
                 if e in df["Event"].unique()]

COUNTRY_CONTINENT = {
    "FRA":"Europe","GBR":"Europe","DEU":"Europe","ITA":"Europe","ESP":"Europe",
    "NLD":"Europe","BEL":"Europe","CHE":"Europe","AUT":"Europe","SWE":"Europe",
    "NOR":"Europe","DNK":"Europe","FIN":"Europe","PRT":"Europe","POL":"Europe",
    "CZE":"Europe","HUN":"Europe","ROU":"Europe","GRC":"Europe","TUR":"Europe",
    "RUS":"Europe","UKR":"Europe","IRL":"Europe","LUX":"Europe","SVK":"Europe",
    "USA":"North America","CAN":"North America","MEX":"North America",
    "BRA":"South America","ARG":"South America","COL":"South America","CHL":"South America",
    "AUS":"Oceania","NZL":"Oceania",
    "ZAF":"Africa","KEN":"Africa","MAR":"Africa","EGY":"Africa",
    "JPN":"Asia","CHN":"Asia","KOR":"Asia","SGP":"Asia","IND":"Asia",
    "ARE":"Asia","SAU":"Asia","QAT":"Asia","ISR":"Asia",
}
df["Continent"] = df["Country"].map(COUNTRY_CONTINENT).fillna("Other")

# Pré-calcul des athlètes présents dans plusieurs événements (par nom normalisé)
_name_event_counts = (df.groupby(df["Name"].str.strip().str.upper())["Event"]
                        .nunique())
MULTI_EVENT_NAMES = set(_name_event_counts[_name_event_counts > 1].index)

def build_athlete_options(d):
    """Construit les options du dropdown — une seule ligne par équipe/duo."""
    # Pour les doublons de team, garder seulement Athlete_Position == 1 (ou la première ligne)
    if "Team_Name" in d.columns and "Athlete_Position" in d.columns:
        solo = d[d["Team_Name"].isna() | (d["Team_Name"] == "")]
        team = d[d["Team_Name"].notna() & (d["Team_Name"] != "")]
        team = team[team["Athlete_Position"] == 1]
        d_dedup = pd.concat([solo, team]).sort_values("Rank")
    else:
        d_dedup = d

    options = []
    for idx, r in d_dedup.iterrows():
        name_upper = str(r["Name"]).strip().upper()
        is_multi   = name_upper in MULTI_EVENT_NAMES
        event_str  = f" · {r['Event']}" if is_multi and pd.notna(r.get("Event")) else ""
        country    = r["Country"] if r["Country"] else ""
        label = (f"#{r['Rank']} {r['Display_Name']}{event_str} "
                 f"({r.get('Category','')}"
                 f"{', ' + country if country else ''}"
                 f", {r['Age_Group']})")
        options.append({"label": label, "value": idx})
    return options

def sec_to_mmss(s):
    if pd.isna(s): return "N/A"
    s = int(s)
    if s >= 3600:
        h = s // 3600; m = (s % 3600) // 60; sec = s % 60
        return f"{h}h{m:02d}:{sec:02d}"
    return f"{s//60}:{s%60:02d}"

def sample_df(d, n=5000):
    return d.sample(n, random_state=42) if len(d) > n else d

# ── Thème ─────────────────────────────────────────────────────────────────────
ACCENT  = "#E8472B"
BG      = "#0F1117"
CARD_BG = "#1A1D27"
TEXT    = "#F0F2F8"
SUBTEXT = "#9CA3AF"
PLOTBG  = "#13151F"
GRID    = "#2A2D3A"
COLORS  = px.colors.qualitative.Vivid
GREEN   = "#10B981"
BLUE    = "#3B82F6"
ORANGE  = "#F59E0B"

CHART_LAYOUT = dict(
    paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT),
)

# ── Filtres pré-calculés ──────────────────────────────────────────────────────
events_sorted     = list(df["Event"].dropna().unique()) if "Event" in df.columns else []
categories_sorted = sorted(df["Category"].dropna().unique()) if "Category" in df.columns else []
countries_sorted  = sorted(df["Country"].dropna().unique())
age_groups_sorted = sorted(df["Age_Group"].dropna().unique())
TOTAL_ATHLETES    = len(df)

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Hyrox Season 8 – All Events",
    assets_folder="assets",
)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html lang="en" translate="no">
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <meta name="google" content="notranslate">
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
'''

# ── Helpers UI ────────────────────────────────────────────────────────────────
_dd_style = {"backgroundColor": CARD_BG, "color": TEXT, "border": f"1px solid {GRID}"}

def kpi_card(icon, title, value, sub=""):
    return dbc.Card([dbc.CardBody([
        html.Div([
            html.I(className=f"bi {icon} fs-3", style={"color": ACCENT}),
            html.Div([
                html.Div(value, style={"fontSize": "1.6rem", "fontWeight": "700", "color": TEXT}),
                html.Div(title, style={"fontSize": ".75rem", "color": SUBTEXT,
                                       "textTransform": "uppercase", "letterSpacing": ".08em"}),
                html.Div(sub, style={"fontSize": ".7rem", "color": SUBTEXT}) if sub else None,
            ], style={"marginLeft": "12px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ])], style={"backgroundColor": CARD_BG, "border": f"1px solid {GRID}", "borderRadius": "10px"})

def card(children, **kwargs):
    return dbc.Card(dbc.CardBody(children),
                    style={"backgroundColor": CARD_BG, "border": f"1px solid {GRID}",
                           "borderRadius": "10px", **kwargs})

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "Inter, sans-serif"},
    children=[
        # Store léger : seulement les paramètres de filtre (pas les 166k indices)
        dcc.Store(id="filtered-store", data={"events": [], "cats": [], "countries": [], "ags": [], "topn": TOTAL_ATHLETES}),
        dcc.Download(id="download-card"),

        # Header
        html.Div([
            html.Div([
                html.Span("⚡", style={"fontSize": "1.6rem", "marginRight": "10px"}),
                html.Span("HYROX Season 8", style={"fontSize": "1.5rem", "fontWeight": "800", "color": TEXT}),
                html.Span(" · All Events 2025-2026", style={"fontSize": "1.1rem", "color": SUBTEXT}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(
                f"{TOTAL_ATHLETES:,} athlètes · {len(events_sorted)} événements · 14 catégories".replace(",", " "),
                style={"color": SUBTEXT, "fontSize": ".8rem"}
            ),
        ], style={
            "backgroundColor": CARD_BG, "padding": "16px 32px",
            "borderBottom": f"2px solid {ACCENT}",
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        }),

        html.Div(style={"padding": "24px 32px"}, children=[

            # Filtres
            dbc.Row([
                dbc.Col([
                    html.Label("Événement", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="filter-event",
                                 options=[{"label": e, "value": e} for e in events_sorted],
                                 multi=True, placeholder="Tous les événements", style=_dd_style),
                ], md=3),
                dbc.Col([
                    html.Label("Catégorie", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="filter-category",
                                 options=[{"label": c, "value": c} for c in categories_sorted],
                                 multi=True, placeholder="Toutes les catégories", style=_dd_style),
                ], md=3),
                dbc.Col([
                    html.Label("Pays", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="filter-country",
                                 options=[{"label": c, "value": c} for c in countries_sorted],
                                 multi=True, placeholder="Tous les pays", style=_dd_style),
                ], md=2),
                dbc.Col([
                    html.Label("Tranche d'âge", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="filter-ag",
                                 options=[{"label": a, "value": a} for a in age_groups_sorted],
                                 multi=True, placeholder="Toutes les tranches", style=_dd_style),
                ], md=2),
                dbc.Col([
                    html.Label("Top N", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Slider(id="filter-topn", min=10, max=TOTAL_ATHLETES, step=50,
                               value=TOTAL_ATHLETES,
                               marks={10: "10", 1000: "1k", 10000: "10k", 50000: "50k",
                                      TOTAL_ATHLETES: "All"},
                               tooltip={"placement": "bottom"},
                               updatemode="mouseup",
                               className="mt-1"),
                ], md=2),
            ], className="mb-4 p-3",
               style={"backgroundColor": CARD_BG, "borderRadius": "10px", "border": f"1px solid {GRID}"}),

            # KPIs
            html.Div(id="kpi-row", className="mb-4"),

            # Tabs
            dcc.Tabs(id="tabs", value="overview", children=[
                dcc.Tab(label="🏁 Vue d'ensemble",   value="overview",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="💪 Stations Workout", value="workouts",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="🏃 Running",          value="running",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="🌍 Géographie",       value="geo",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="👥 Démographie",      value="demo",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="🔍 Profil Athlète",   value="athlete",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="🧠 Analyse Coach",    value="coach",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="📈 Évolution Saison",  value="season",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="📋 Classement",       value="ranking",
                        style={"backgroundColor": BG, "color": SUBTEXT},
                        selected_style={"backgroundColor": CARD_BG, "color": TEXT,
                                        "borderTop": f"2px solid {ACCENT}"}),
            ], style={"marginBottom": "20px"}),

            dcc.Loading(
                id="loading-tab",
                type="dot",
                color=ACCENT,
                children=html.Div(id="tab-content"),
            ),
        ]),
    ]
)

# ── Cache de filtrage (evite de recalculer à chaque callback) ─────────────────
@lru_cache(maxsize=128)
def _cached_filter(events_t, cats_t, countries_t, ags_t, topn):
    mask = pd.Series(True, index=df.index)
    if events_t:    mask &= df["Event"].isin(events_t)
    if cats_t:      mask &= df["Category"].isin(cats_t)
    if countries_t: mask &= df["Country"].isin(countries_t)
    if ags_t:       mask &= df["Age_Group"].isin(ags_t)
    idx = df.index[mask]
    if topn and topn < len(idx):
        idx = idx[:topn]
    return idx.tolist()

def get_filtered_df(store):
    if store is None:
        return df
    ev  = tuple(sorted(store.get("events",   []) or []))
    cat = tuple(sorted(store.get("cats",     []) or []))
    co  = tuple(sorted(store.get("countries",[]) or []))
    ag  = tuple(sorted(store.get("ags",      []) or []))
    tn  = store.get("topn", TOTAL_ATHLETES) or TOTAL_ATHLETES
    return df.loc[_cached_filter(ev, cat, co, ag, tn)]

# ── CB1 : Store (stocke les paramètres, pas les indices) ──────────────────────
@app.callback(
    Output("filtered-store", "data"),
    [Input("filter-event", "value"), Input("filter-category", "value"),
     Input("filter-country", "value"), Input("filter-ag", "value"),
     Input("filter-topn", "value")]
)
def update_store(events, categories, countries, ags, topn):
    return {"events": events or [], "cats": categories or [],
            "countries": countries or [], "ags": ags or [],
            "topn": topn or TOTAL_ATHLETES}

# ── CB2 : KPIs ────────────────────────────────────────────────────────────────
@app.callback(Output("kpi-row", "children"), Input("filtered-store", "data"))
def update_kpis(store):
    d = get_filtered_df(store)
    n = len(d)
    if n == 0:
        return dbc.Row([dbc.Col(kpi_card("bi-people-fill", "Athlètes", "0", "0 pays"), md=3)])
    best_idx  = d["Total_sec"].idxmin()
    best      = sec_to_mmss(d.at[best_idx, "Total_sec"])
    best_name = d.at[best_idx, "Display_Name"]
    avg = sec_to_mmss(d["Total_sec"].mean())
    med = sec_to_mmss(d["Total_sec"].median())
    n_countries = d["Country"].nunique()
    return dbc.Row([
        dbc.Col(kpi_card("bi-people-fill",    "Athlètes",       str(n),  f"{n_countries} pays"), md=3),
        dbc.Col(kpi_card("bi-trophy-fill",    "Meilleur temps", best,    best_name), md=3),
        dbc.Col(kpi_card("bi-bar-chart-fill", "Temps moyen",    avg,     ""), md=3),
        dbc.Col(kpi_card("bi-activity",       "Temps médian",   med,     ""), md=3),
    ], className="g-3")

# ── CB3 : Onglets ─────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"), Input("filtered-store", "data")]
)
def render_tab(tab, store):
    d = get_filtered_df(store)
    if len(d) == 0:
        return html.Div("Aucune donnée pour ces filtres.", style={"color": SUBTEXT, "padding": "40px"})

    # ── Vue d'ensemble ────────────────────────────────────────────────────────
    if tab == "overview":
        fig_hist = go.Figure(go.Histogram(
            x=d["Total_min"], nbinsx=60, marker_color=ACCENT, opacity=0.85))
        med = d["Total_min"].median()
        fig_hist.add_vline(x=med, line_dash="dash", line_color=BLUE,
                           annotation_text=f"Médiane {sec_to_mmss(med*60)}",
                           annotation_font_color=BLUE)
        fig_hist.update_layout(title="Distribution des temps finaux",
                               xaxis_title="Temps (min)", yaxis_title="Athlètes", **CHART_LAYOUT)

        ds = sample_df(d, 5000)
        fig_scatter = px.scatter(ds, x="Runs_min", y="Workouts_min", color="Age_Group",
                                 hover_name="Display_Name",
                                 labels={"Runs_min": "Running (min)", "Workouts_min": "Workout (min)"},
                                 title=f"Running vs Workout ({len(ds):,} pts)".replace(",", " "),
                                 color_discrete_sequence=COLORS)
        fig_scatter.update_layout(**CHART_LAYOUT)
        fig_scatter.update_traces(marker=dict(size=4, opacity=0.6))

        top20 = d.head(20).copy()
        top20["label"] = top20["Display_Name"].str.split(",").str[0].str.split().str[-1] \
                         + " (" + top20["Country"] + ")"
        fig_top20 = go.Figure(go.Bar(
            y=top20["label"][::-1], x=top20["Total_min"][::-1], orientation="h",
            marker_color=ACCENT, text=top20["Finish_Time"][::-1], textposition="outside"))
        fig_top20.update_layout(title="Top 20 athlètes", xaxis_title="Temps (min)",
                                height=550, **CHART_LAYOUT)

        fig_donut = go.Figure(go.Pie(
            labels=["Running", "Workout", "Roxzone"],
            values=[d["Runs_min"].mean(), d["Workouts_min"].mean(), d["Roxzone_min"].mean()],
            hole=0.6, marker_colors=[ACCENT, BLUE, GREEN], textinfo="label+percent"))
        fig_donut.update_layout(title="Décomposition moyenne", **CHART_LAYOUT)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_hist,    config={"displayModeBar": False})), md=8),
                dbc.Col(card(dcc.Graph(figure=fig_donut,   config={"displayModeBar": False})), md=4),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_top20,   config={"displayModeBar": False})), md=5),
                dbc.Col(card(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False})), md=7),
            ], className="g-3"),
        ])

    # ── Stations Workout ──────────────────────────────────────────────────────
    elif tab == "workouts":
        avgs = [d[c].mean() for c in WORKOUT_COLS]
        fig_bar = go.Figure(go.Bar(
            x=WORKOUT_LABELS, y=[a/60 for a in avgs], marker_color=ACCENT,
            text=[sec_to_mmss(a) for a in avgs], textposition="outside"))
        fig_bar.update_layout(title="Temps moyen par station", yaxis_title="Temps (min)", **CHART_LAYOUT)

        fig_box = go.Figure()
        for col, label in zip(WORKOUT_COLS, WORKOUT_LABELS):
            fig_box.add_trace(go.Box(y=d[col].dropna()/60, name=label,
                                     marker_color=ACCENT, line_color=ACCENT, boxmean=True))
        fig_box.update_layout(title="Distribution par station", yaxis_title="Temps (min)",
                               showlegend=False, **CHART_LAYOUT)

        corr_data = d[WORKOUT_COLS].copy()
        corr_data.columns = WORKOUT_LABELS
        fig_corr = px.imshow(corr_data.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Corrélations entre stations")
        fig_corr.update_layout(**CHART_LAYOUT)

        score_avgs = [d[c].mean() for c in SCORE_COLS if c in d.columns]
        fig_score = go.Figure(go.Bar(
            x=WORKOUT_LABELS[:len(score_avgs)], y=score_avgs,
            marker=dict(color=score_avgs, colorscale=[[0,"#EF4444"],[0.5,ORANGE],[1,GREEN]],
                        showscale=True, cmin=0, cmax=100),
            text=[f"{s:.1f}" for s in score_avgs], textposition="outside"))
        score_layout = {**CHART_LAYOUT, "yaxis": {**CHART_LAYOUT.get("yaxis",{}), "range": [0,105]}}
        fig_score.update_layout(title="Score moyen par station", **score_layout)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_bar,   config={"displayModeBar": False})), md=6),
                dbc.Col(card(dcc.Graph(figure=fig_score, config={"displayModeBar": False})), md=6),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_box,  config={"displayModeBar": False})), md=8),
                dbc.Col(card(dcc.Graph(figure=fig_corr, config={"displayModeBar": False})), md=4),
            ], className="g-3"),
        ])

    # ── Running ───────────────────────────────────────────────────────────────
    elif tab == "running":
        run_avgs = [d[c].mean() for c in RUN_COLS if c in d.columns]
        fig_run = go.Figure(go.Bar(
            x=RUN_LABELS[:len(run_avgs)], y=[a/60 for a in run_avgs],
            marker_color=BLUE, text=[sec_to_mmss(a) for a in run_avgs], textposition="outside"))
        fig_run.update_layout(title="Temps moyen par run", yaxis_title="Temps (min)", **CHART_LAYOUT)

        top10    = d.head(10)
        med_runs = [d[c].median()/60        for c in RUN_COLS if c in d.columns]
        top_runs = [top10[c].mean()/60      for c in RUN_COLS if c in d.columns]
        bot_runs = [d.tail(100)[c].mean()/60 for c in RUN_COLS if c in d.columns]
        fig_lines = go.Figure()
        fig_lines.add_trace(go.Scatter(x=RUN_LABELS[:len(med_runs)], y=top_runs, mode="lines+markers",
                                       name="Top 10", line=dict(color=ACCENT, width=3)))
        fig_lines.add_trace(go.Scatter(x=RUN_LABELS[:len(med_runs)], y=med_runs, mode="lines+markers",
                                       name="Médiane", line=dict(color=BLUE, width=2)))
        fig_lines.add_trace(go.Scatter(x=RUN_LABELS[:len(bot_runs)], y=bot_runs, mode="lines+markers",
                                       name="Bas 100", line=dict(color=SUBTEXT, width=2, dash="dot")))
        fig_lines.update_layout(title="Top 10 / Médiane / Bas 100",
                                yaxis_title="Temps (min)", **CHART_LAYOUT)

        cols_heat   = [c for c in RUN_COLS + WORKOUT_COLS if c in d.columns]
        labels_heat = RUN_LABELS[:sum(c in d.columns for c in RUN_COLS)] + \
                      WORKOUT_LABELS[:sum(c in d.columns for c in WORKOUT_COLS)]
        corr_rw = d[cols_heat].corr()
        corr_rw.index = corr_rw.columns = labels_heat[:len(corr_rw)]
        fig_heatmap = px.imshow(corr_rw, text_auto=".2f", color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1, title="Corrélations Runs × Stations")
        fig_heatmap.update_layout(**CHART_LAYOUT)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_run,     config={"displayModeBar": False})), md=5),
                dbc.Col(card(dcc.Graph(figure=fig_lines,   config={"displayModeBar": False})), md=7),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_heatmap, config={"displayModeBar": False})), md=12),
            ], className="g-3"),
        ])

    # ── Géographie ────────────────────────────────────────────────────────────
    elif tab == "geo":
        d_geo = d[d["Country"].astype(str).str.strip() != ""].copy()
        d_geo["Country"] = d_geo["Country"].astype(str)
        by_country = (d_geo.groupby("Country", sort=False, observed=True)
                       .agg(Athletes=("Name","count"),
                            Avg_min=("Total_min","mean"),
                            Best_min=("Total_min","min"),
                            Median_min=("Total_min","median"))
                       .round(2).reset_index()
                       .sort_values("Athletes", ascending=False))

        fig_pays = go.Figure(go.Bar(
            y=by_country.head(20)["Country"][::-1],
            x=by_country.head(20)["Athletes"][::-1],
            orientation="h", marker_color=ACCENT,
            text=by_country.head(20)["Athletes"][::-1], textposition="outside"))
        fig_pays.update_layout(title="Top 20 pays – Athlètes", height=500, **CHART_LAYOUT)

        fig_scat = px.scatter(by_country[by_country["Athletes"] > 0],
                              x="Athletes", y="Avg_min", text="Country",
                              size="Athletes", color="Avg_min",
                              color_continuous_scale="RdYlGn_r",
                              labels={"Avg_min": "Temps moyen (min)"},
                              title="Volume vs Performance")
        fig_scat.update_layout(**CHART_LAYOUT)
        fig_scat.update_traces(textposition="top center", marker=dict(sizemin=6))

        top5 = by_country.head(5)["Country"].tolist()
        d5   = d_geo[d_geo["Country"].isin(top5)]
        fig_violin = go.Figure()
        for c in top5:
            fig_violin.add_trace(go.Violin(
                y=d5[d5["Country"] == c]["Total_min"].dropna(),
                name=c, box_visible=True, meanline_visible=True, line_color=ACCENT))
        fig_violin.update_layout(title="Distribution – Top 5 pays",
                                  yaxis_title="Temps (min)", **CHART_LAYOUT)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_pays,   config={"displayModeBar": False})), md=5),
                dbc.Col(card(dcc.Graph(figure=fig_scat,   config={"displayModeBar": False})), md=7),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_violin, config={"displayModeBar": False})), md=12),
            ], className="g-3"),
        ])

    # ── Démographie ───────────────────────────────────────────────────────────
    elif tab == "demo":
        by_ag = (d.groupby("Age_Group", sort=False)
                  .agg(Athletes=("Name","count"),
                       Avg_min=("Total_min","mean"),
                       Best_min=("Total_min","min"),
                       Median_min=("Total_min","median"),
                       Avg_Run_min=("Runs_min","mean"),
                       Avg_Work_min=("Workouts_min","mean"))
                  .round(2).reset_index())

        fig_ag_n = go.Figure(go.Bar(
            x=by_ag["Age_Group"], y=by_ag["Athletes"],
            marker_color=ACCENT, text=by_ag["Athletes"], textposition="outside"))
        fig_ag_n.update_layout(title="Athlètes par tranche d'âge", **CHART_LAYOUT)

        fig_ag_time = go.Figure()
        fig_ag_time.add_trace(go.Scatter(x=by_ag["Age_Group"], y=by_ag["Avg_min"],
                                         name="Moyen", mode="lines+markers",
                                         line=dict(color=ACCENT, width=3)))
        fig_ag_time.add_trace(go.Scatter(x=by_ag["Age_Group"], y=by_ag["Best_min"],
                                         name="Meilleur", mode="lines+markers",
                                         line=dict(color=GREEN, width=2, dash="dash")))
        fig_ag_time.add_trace(go.Scatter(x=by_ag["Age_Group"], y=by_ag["Median_min"],
                                         name="Médian", mode="lines+markers",
                                         line=dict(color=BLUE, width=2, dash="dot")))
        fig_ag_time.update_layout(title="Temps par tranche d'âge",
                                   yaxis_title="Temps (min)", **CHART_LAYOUT)

        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(name="Running", x=by_ag["Age_Group"],
                                     y=by_ag["Avg_Run_min"], marker_color=BLUE))
        fig_stacked.add_trace(go.Bar(name="Workout", x=by_ag["Age_Group"],
                                     y=by_ag["Avg_Work_min"], marker_color=ACCENT))
        fig_stacked.update_layout(barmode="stack", title="Running/Workout par âge",
                                   yaxis_title="Temps (min)", **CHART_LAYOUT)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_ag_n,      config={"displayModeBar": False})), md=4),
                dbc.Col(card(dcc.Graph(figure=fig_ag_time,   config={"displayModeBar": False})), md=8),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_stacked,   config={"displayModeBar": False})), md=12),
            ], className="g-3"),
        ])

    # ── Profil Athlète ────────────────────────────────────────────────────────
    elif tab == "athlete":
        athlete_options = build_athlete_options(d)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label(f"Sélectionner un athlète ({len(athlete_options):,} athlètes)".replace(",", " "),
                               style={"color": SUBTEXT}),
                    dcc.Dropdown(id="athlete-select", options=athlete_options,
                                 value=athlete_options[0]["value"] if athlete_options else None,
                                 style=_dd_style),
                ], md=6),
                dbc.Col([
                    html.Label("Comparer avec", style={"color": SUBTEXT}),
                    dcc.Dropdown(id="athlete-compare",
                                 options=[{"label": "Médiane", "value": "median"},
                                          {"label": "Top 10%", "value": "top10"}],
                                 value="median", style=_dd_style),
                ], md=3),
            ], className="mb-3"),
            html.Div(id="athlete-content"),
        ])

    # ── Analyse Coach ─────────────────────────────────────────────────────────
    elif tab == "coach":
        coach_options = build_athlete_options(d)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label(f"Sélectionner un athlète pour l'analyse ({len(coach_options):,} athlètes)".replace(",", " "),
                               style={"color": SUBTEXT}),
                    dcc.Dropdown(id="coach-select", options=coach_options,
                                 value=coach_options[0]["value"] if coach_options else None,
                                 style=_dd_style),
                ], md=7),
            ], className="mb-3"),
            html.Div(id="coach-content"),
        ])

    # ── Classement ────────────────────────────────────────────────────────────
    elif tab == "ranking":
        cols_show = ["Rank","Rank_AG","Event","Category","Display_Name",
                     "Country","Age_Group","Finish_Time",
                     "Runs_Total_sec","Workouts_Total_sec","Roxzone_sec"]
        d_table = d[[c for c in cols_show if c in d.columns]].head(1000).copy()
        d_table["Running"] = d_table["Runs_Total_sec"].apply(sec_to_mmss)
        d_table["Workout"] = d_table["Workouts_Total_sec"].apply(sec_to_mmss)
        d_table["Roxzone"] = d_table["Roxzone_sec"].apply(sec_to_mmss)
        d_table.drop(columns=["Runs_Total_sec","Workouts_Total_sec","Roxzone_sec"],
                     inplace=True, errors="ignore")
        d_table.rename(columns={"Display_Name": "Nom(s)"}, inplace=True)

        return card(dash_table.DataTable(
            data=d_table.to_dict("records"),
            columns=[{"name": c, "id": c} for c in d_table.columns],
            page_size=25, sort_action="native", filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": CARD_BG, "color": TEXT,
                        "border": f"1px solid {GRID}", "fontSize": "13px",
                        "textAlign": "left", "padding": "8px"},
            style_header={"backgroundColor": BG, "color": ACCENT,
                          "fontWeight": "bold", "border": f"1px solid {GRID}"},
            style_data_conditional=[
                {"if": {"row_index": 0}, "backgroundColor": "#2D1F1A", "color": "#F97316"},
            ],
        ))

    # ── Évolution Saison ──────────────────────────────────────────────────────
    elif tab == "season":
        events_avail = [e for e in EVENTS_CHRONO if e in df["Event"].unique()]
        return html.Div([
            # Section 1 : Vue macro
            html.H5("📊 Vue macro de la saison", style={"color": ACCENT, "marginBottom": "16px",
                                                         "textTransform": "uppercase", "letterSpacing": ".08em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Catégories à afficher", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="season-cat-filter",
                                 options=[{"label": c, "value": c} for c in categories_sorted],
                                 multi=True, placeholder="Toutes (max 5 pour lisibilité)",
                                 value=["HYROX MEN","HYROX WOMEN","HYROX PRO MEN"],
                                 style=_dd_style),
                ], md=6),
            ], className="mb-3"),
            html.Div(id="season-macro"),
            html.Hr(style={"borderColor": GRID, "margin": "32px 0"}),

            # Section 2 : Recherche athlète
            html.H5("🔍 Suivi d'un athlète sur la saison",
                    style={"color": ACCENT, "marginBottom": "16px",
                           "textTransform": "uppercase", "letterSpacing": ".08em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Nom de l'athlète (tape pour chercher, insensible aux accents)",
                               style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="season-athlete-search",
                                 options=[], value=None,
                                 placeholder="Ex: Gregoire, Müller, O'Brien...",
                                 searchable=True, clearable=True,
                                 style=_dd_style),
                ], md=6),
            ], className="mb-3"),
            html.Div(id="season-athlete-result"),
            html.Hr(style={"borderColor": GRID, "margin": "32px 0"}),

            # Section 3 : Benchmark inter-events
            html.H5("⚖️ Benchmark inter-événements",
                    style={"color": ACCENT, "marginBottom": "16px",
                           "textTransform": "uppercase", "letterSpacing": ".08em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Événement A", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="bench-event-a",
                                 options=[{"label": e, "value": e} for e in events_avail],
                                 value=events_avail[0] if events_avail else None,
                                 style=_dd_style),
                ], md=3),
                dbc.Col([
                    html.Label("Événement B", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="bench-event-b",
                                 options=[{"label": e, "value": e} for e in events_avail],
                                 value=events_avail[1] if len(events_avail) > 1 else None,
                                 style=_dd_style),
                ], md=3),
                dbc.Col([
                    html.Label("Catégorie", style={"color": SUBTEXT, "fontSize": ".75rem"}),
                    dcc.Dropdown(id="bench-category",
                                 options=[{"label": c, "value": c} for c in categories_sorted],
                                 value="HYROX MEN", style=_dd_style),
                ], md=3),
            ], className="mb-3"),
            html.Div(id="season-benchmark"),
        ])

    return html.Div("Chargement...", style={"color": SUBTEXT})


# ── CB4 : Profil athlète ──────────────────────────────────────────────────────
@app.callback(
    Output("athlete-content", "children"),
    [Input("athlete-select", "value"), Input("athlete-compare", "value")],
    State("filtered-store", "data"),
    prevent_initial_call=True
)
def update_athlete_profile(idx, compare_mode, store):
    if idx is None:
        return html.Div("Sélectionnez un athlète", style={"color": SUBTEXT})

    d   = get_filtered_df(store)
    row = df.loc[idx]

    if compare_mode == "median":
        ref      = d[WORKOUT_COLS].median()
        ref_runs = d[[c for c in RUN_COLS if c in d.columns]].median()
        ref_label = "Médiane"
    else:
        q10   = d["Total_sec"].quantile(0.10)
        top_d = d[d["Total_sec"] <= q10]
        ref      = top_d[WORKOUT_COLS].mean() if len(top_d) else d[WORKOUT_COLS].median()
        ref_runs = (top_d[[c for c in RUN_COLS if c in top_d.columns]].mean()
                    if len(top_d) else d[[c for c in RUN_COLS if c in d.columns]].median())
        ref_label = "Top 10%"

    display_name = row.get("Team_Name") if pd.notna(row.get("Team_Name")) else row["Name"]
    short_name   = str(display_name).split(",")[0].split()[-1]
    rank         = row.get("Rank", "?")
    rank_ag      = row.get("Rank_AG", "?")
    pct          = (1 - (rank - 1) / len(d)) * 100 if isinstance(rank, (int, float)) else 0

    ath_vals = [row[c]/60 if pd.notna(row.get(c)) else 0 for c in WORKOUT_COLS]
    ref_vals = [ref[c]/60 for c in WORKOUT_COLS]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=ath_vals + [ath_vals[0]], theta=WORKOUT_LABELS + [WORKOUT_LABELS[0]],
        fill="toself", name=short_name,
        line=dict(color=ACCENT), fillcolor="rgba(232,71,43,0.2)"))
    fig_radar.add_trace(go.Scatterpolar(
        r=ref_vals + [ref_vals[0]], theta=WORKOUT_LABELS + [WORKOUT_LABELS[0]],
        fill="toself", name=ref_label,
        line=dict(color=BLUE), fillcolor="rgba(59,130,246,0.15)"))
    fig_radar.update_layout(
        title=f"Radar Workout — {display_name} vs {ref_label}",
        polar=dict(radialaxis=dict(visible=True, gridcolor=GRID, color=SUBTEXT),
                   bgcolor=PLOTBG, angularaxis=dict(color=SUBTEXT)),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ["xaxis","yaxis"]})

    run_cols_avail = [c for c in RUN_COLS if c in df.columns]
    ath_run = [row[c]/60 if pd.notna(row.get(c)) else 0 for c in run_cols_avail]
    ref_run = [ref_runs[c]/60 for c in run_cols_avail]
    fig_runs = go.Figure()
    fig_runs.add_trace(go.Bar(name=short_name, x=RUN_LABELS[:len(ath_run)], y=ath_run,
                              marker_color=ACCENT,
                              text=[sec_to_mmss(v*60) for v in ath_run], textposition="outside"))
    fig_runs.add_trace(go.Bar(name=ref_label, x=RUN_LABELS[:len(ref_run)], y=ref_run,
                              marker_color=BLUE,
                              text=[sec_to_mmss(v*60) for v in ref_run], textposition="outside"))
    fig_runs.update_layout(barmode="group", title="Splits Running",
                            yaxis_title="Temps (min)", **CHART_LAYOUT)

    medians = {col: d[col].median() for col in WORKOUT_COLS if col in d.columns}
    score_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.Div(sec_to_mmss(row.get(WORKOUT_COLS[i])) if pd.notna(row.get(WORKOUT_COLS[i])) else "–",
                     style={"fontSize": "1.4rem", "fontWeight": "700",
                            "color": (GREEN if pd.notna(row.get(WORKOUT_COLS[i])) and
                                      row.get(WORKOUT_COLS[i]) <= medians.get(WORKOUT_COLS[i], 99999)
                                      else ORANGE if pd.notna(row.get(WORKOUT_COLS[i])) and
                                      row.get(WORKOUT_COLS[i]) <= medians.get(WORKOUT_COLS[i], 99999) * 1.1
                                      else ACCENT)}),
            html.Div(lbl, style={"fontSize": ".65rem", "color": SUBTEXT}),
        ])], style={"backgroundColor": BG, "border": f"1px solid {GRID}", "textAlign": "center"}),
        md=3) for i, lbl in enumerate(WORKOUT_LABELS)
    ], className="g-2")

    summary_card = dbc.Card([dbc.CardBody([
        html.H4(str(display_name), style={"color": TEXT, "fontWeight": "800"}),
        html.Div(f"{row.get('Event','')} · {row.get('Category','')} · "
                 f"{row.get('Age_Group','')} · {row.get('Country','')}",
                 style={"color": SUBTEXT, "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([html.Div(row.get("Finish_Time","–"),
                              style={"fontSize": "2rem", "fontWeight": "800", "color": ACCENT}),
                     html.Div("Temps final", style={"color": SUBTEXT, "fontSize": ".75rem"})]),
            dbc.Col([html.Div(f"#{rank}",
                              style={"fontSize": "1.8rem", "fontWeight": "700", "color": TEXT}),
                     html.Div("Classement général", style={"color": SUBTEXT, "fontSize": ".75rem"})]),
            dbc.Col([html.Div(f"#{rank_ag}",
                              style={"fontSize": "1.8rem", "fontWeight": "700", "color": TEXT}),
                     html.Div("Classement AG", style={"color": SUBTEXT, "fontSize": ".75rem"})]),
            dbc.Col([html.Div(f"Top {100-pct:.1f}%",
                              style={"fontSize": "1.4rem", "fontWeight": "700", "color": GREEN}),
                     html.Div("Percentile", style={"color": SUBTEXT, "fontSize": ".75rem"})]),
        ]),
        html.Hr(style={"borderColor": GRID}),
        html.Div("Temps par station",
                 style={"color": SUBTEXT, "marginBottom": "8px",
                        "fontSize": ".75rem", "textTransform": "uppercase"}),
        score_cards,
        html.Hr(style={"borderColor": GRID}),
        dbc.Row([
            dbc.Col(dbc.Button("📸 Générer ma carte",
                               id="btn-card", color="danger", className="mt-2")),
        ]),
    ])], style={"backgroundColor": CARD_BG, "border": f"1px solid {GRID}", "borderRadius": "10px"})

    return html.Div([
        summary_card,
        html.Div(style={"height": "16px"}),
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig_radar, config={"displayModeBar": False})), md=5),
            dbc.Col(card(dcc.Graph(figure=fig_runs,  config={"displayModeBar": False})), md=7),
        ], className="g-3"),
    ])


# ── CB5 : Génération carte PNG ────────────────────────────────────────────────
@app.callback(
    Output("download-card", "data"),
    Input("btn-card", "n_clicks"),
    [State("athlete-select", "value"), State("filtered-store", "data")],
    prevent_initial_call=True
)
def generate_card(n_clicks, idx, store):
    if not n_clicks or idx is None:
        return None

    d   = get_filtered_df(store)
    row = df.loc[idx]
    display_name = row.get("Team_Name") if pd.notna(row.get("Team_Name")) else row["Name"]
    rank         = row.get("Rank", "?")
    rank_ag      = row.get("Rank_AG", "?")
    finish_time  = row.get("Finish_Time", "–")
    pct          = (1 - (rank - 1) / len(d)) * 100 if isinstance(rank, (int, float)) else 0

    # Stations : trier par temps pour trouver les 3 meilleures et la pire
    station_times = []
    for lbl, col in zip(WORKOUT_LABELS, WORKOUT_COLS):
        val = row.get(col)
        if pd.notna(val):
            station_times.append((lbl, val))
    station_times_sorted = sorted(station_times, key=lambda x: x[1])
    best3 = station_times_sorted[:3]
    worst = station_times_sorted[-1] if station_times_sorted else None

    # Construction figure 1080x1080
    fig = go.Figure()
    fig.update_layout(
        width=1080, height=1080,
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        shapes=[
            # Liseré rouge gauche
            dict(type="rect", xref="paper", yref="paper",
                 x0=0, y0=0, x1=0.012, y1=1,
                 fillcolor=ACCENT, line_color=ACCENT),
            # Ligne séparatrice
            dict(type="line", xref="paper", yref="paper",
                 x0=0.06, y0=0.82, x1=0.94, y1=0.82,
                 line=dict(color=ACCENT, width=2)),
            dict(type="line", xref="paper", yref="paper",
                 x0=0.06, y0=0.18, x1=0.94, y1=0.18,
                 line=dict(color=GRID, width=1)),
        ],
        annotations=[
            # Header
            dict(text="⚡ HYROX", xref="paper", yref="paper", x=0.08, y=0.94,
                 showarrow=False, font=dict(size=28, color=ACCENT, family="Inter"),
                 xanchor="left"),
            dict(text=f"{row.get('Event','NICE 2026')} · {row.get('Category','')}",
                 xref="paper", yref="paper", x=0.08, y=0.90,
                 showarrow=False, font=dict(size=18, color=SUBTEXT), xanchor="left"),
            # Nom
            dict(text=str(display_name).upper(),
                 xref="paper", yref="paper", x=0.08, y=0.80,
                 showarrow=False, font=dict(size=36, color=TEXT, family="Inter"), xanchor="left"),
            # KPIs
            dict(text=finish_time, xref="paper", yref="paper", x=0.15, y=0.68,
                 showarrow=False, font=dict(size=42, color=ACCENT, family="Inter"), xanchor="center"),
            dict(text="TEMPS FINAL", xref="paper", yref="paper", x=0.15, y=0.61,
                 showarrow=False, font=dict(size=13, color=SUBTEXT), xanchor="center"),
            dict(text=f"#{rank}", xref="paper", yref="paper", x=0.40, y=0.68,
                 showarrow=False, font=dict(size=42, color=TEXT, family="Inter"), xanchor="center"),
            dict(text="CLASSEMENT", xref="paper", yref="paper", x=0.40, y=0.61,
                 showarrow=False, font=dict(size=13, color=SUBTEXT), xanchor="center"),
            dict(text=f"#{rank_ag}", xref="paper", yref="paper", x=0.63, y=0.68,
                 showarrow=False, font=dict(size=42, color=TEXT, family="Inter"), xanchor="center"),
            dict(text="RANG AG", xref="paper", yref="paper", x=0.63, y=0.61,
                 showarrow=False, font=dict(size=13, color=SUBTEXT), xanchor="center"),
            dict(text=f"TOP {100-pct:.0f}%", xref="paper", yref="paper", x=0.86, y=0.68,
                 showarrow=False, font=dict(size=42, color=GREEN, family="Inter"), xanchor="center"),
            dict(text="PERCENTILE", xref="paper", yref="paper", x=0.86, y=0.61,
                 showarrow=False, font=dict(size=13, color=SUBTEXT), xanchor="center"),
            # Stations
            dict(text="MEILLEURES STATIONS", xref="paper", yref="paper", x=0.08, y=0.55,
                 showarrow=False, font=dict(size=14, color=SUBTEXT), xanchor="left"),
        ]
    )

    # 3 meilleures stations
    for i, (lbl, val) in enumerate(best3):
        x_pos = 0.18 + i * 0.28
        fig.add_annotation(text=f"✓ {lbl}", xref="paper", yref="paper",
                           x=x_pos, y=0.50, showarrow=False,
                           font=dict(size=16, color=GREEN), xanchor="center")
        fig.add_annotation(text=sec_to_mmss(val), xref="paper", yref="paper",
                           x=x_pos, y=0.45, showarrow=False,
                           font=dict(size=22, color=GREEN, family="Inter"), xanchor="center")

    # Pire station
    if worst:
        fig.add_annotation(text="À TRAVAILLER", xref="paper", yref="paper",
                           x=0.08, y=0.38, showarrow=False,
                           font=dict(size=14, color=SUBTEXT), xanchor="left")
        fig.add_annotation(text=f"⚠ {worst[0]}", xref="paper", yref="paper",
                           x=0.5, y=0.33, showarrow=False,
                           font=dict(size=20, color=ACCENT), xanchor="center")
        fig.add_annotation(text=sec_to_mmss(worst[1]), xref="paper", yref="paper",
                           x=0.5, y=0.27, showarrow=False,
                           font=dict(size=28, color=ACCENT, family="Inter"), xanchor="center")

    # Footer
    fig.add_annotation(text="hyrox-dashboard.com · Saison 8 · 2025-2026",
                       xref="paper", yref="paper", x=0.5, y=0.08,
                       showarrow=False, font=dict(size=14, color=SUBTEXT), xanchor="center")
    fig.add_annotation(text=f"{row.get('Country','')} · {row.get('Age_Group','')}",
                       xref="paper", yref="paper", x=0.5, y=0.04,
                       showarrow=False, font=dict(size=13, color=SUBTEXT), xanchor="center")

    buf = io.BytesIO()
    fig.write_image(buf, format="png", width=1080, height=1080, scale=1)
    encoded = base64.b64encode(buf.getvalue()).decode()
    filename = f"hyrox_{str(display_name).replace(' ','_').replace(',','-')}.png"
    return dict(content=encoded, filename=filename, type="image/png", base64=True)


# ── CB6 : Analyse Coach ───────────────────────────────────────────────────────
@app.callback(
    Output("coach-content", "children"),
    Input("coach-select", "value"),
    State("filtered-store", "data"),
    prevent_initial_call=True
)
def update_coach(idx, store):
    if idx is None:
        return html.Div("Sélectionnez un athlète", style={"color": SUBTEXT})

    d   = get_filtered_df(store)
    row = df.loc[idx]

    display_name = row.get("Team_Name") if pd.notna(row.get("Team_Name")) else row["Name"]
    first_name   = str(display_name).split(",")[0].split()[0]
    rank         = row.get("Rank")
    n            = len(d)
    pct_global   = (1 - (rank - 1) / n) * 100 if isinstance(rank, (int, float)) and n > 0 else None

    # Catégorie de l'athlète pour les médianes de référence
    cat  = row.get("Category", "")
    d_cat = df[df["Category"] == cat] if cat else d

    # ── 1. Diagnostic ─────────────────────────────────────────────────────────
    if pct_global is not None:
        diag_text = (f"{first_name}, tu es dans le Top {100-pct_global:.0f}% global "
                     f"(#{rank} sur {n}) — voici où tu laisses du temps sur la table.")
    else:
        diag_text = f"{first_name}, analyse de ta performance :"

    diag_card = dbc.Alert([
        html.I(className="bi bi-lightning-fill me-2"),
        html.Strong(diag_text)
    ], color="warning", style={"backgroundColor": "#2D2000", "borderColor": ORANGE,
                                "color": TEXT, "fontSize": "1.05rem"})

    # ── 2. Tableau des gains potentiels ───────────────────────────────────────
    cat_medians = {col: d_cat[col].median() for col in WORKOUT_COLS if col in d_cat.columns}
    gains = []
    for lbl, col in zip(WORKOUT_LABELS, WORKOUT_COLS):
        ath_val = row.get(col)
        med_val = cat_medians.get(col)
        if pd.isna(ath_val) or pd.isna(med_val):
            continue
        delta = ath_val - med_val  # positif = plus lent que médiane
        gains.append({"Station": lbl, "Ton temps": ath_val, "Médiane catégorie": med_val,
                      "Écart (s)": delta, "col": col})
    gains_df = pd.DataFrame(gains).sort_values("Écart (s)", ascending=False)

    table_rows = []
    for _, g in gains_df.iterrows():
        ecart = g["Écart (s)"]
        if ecart > 30:
            color = "#3B0000"; text_color = "#FCA5A5"
        elif ecart > 0:
            color = "#2D1A00"; text_color = ORANGE
        else:
            color = "#002B1A"; text_color = GREEN
        gain_str = f"-{sec_to_mmss(abs(ecart))}" if ecart > 0 else f"+{sec_to_mmss(abs(ecart))}"
        table_rows.append(html.Tr([
            html.Td(g["Station"], style={"color": TEXT, "padding": "8px 12px"}),
            html.Td(sec_to_mmss(g["Ton temps"]), style={"color": ACCENT, "padding": "8px 12px", "textAlign":"center"}),
            html.Td(sec_to_mmss(g["Médiane catégorie"]), style={"color": SUBTEXT, "padding": "8px 12px", "textAlign":"center"}),
            html.Td(gain_str, style={"color": text_color, "padding": "8px 12px",
                                     "textAlign":"center", "fontWeight":"700",
                                     "backgroundColor": color, "borderRadius":"4px"}),
        ], style={"borderBottom": f"1px solid {GRID}"}))

    gains_table = dbc.Card([dbc.CardBody([
        html.H6("📊 Tableau des gains potentiels",
                style={"color": ACCENT, "textTransform": "uppercase", "letterSpacing": ".08em"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Station",          style={"color": SUBTEXT, "padding": "8px 12px"}),
                html.Th("Ton temps",        style={"color": SUBTEXT, "padding": "8px 12px", "textAlign":"center"}),
                html.Th("Médiane catégorie",style={"color": SUBTEXT, "padding": "8px 12px", "textAlign":"center"}),
                html.Th("Écart",            style={"color": SUBTEXT, "padding": "8px 12px", "textAlign":"center"}),
            ]), style={"borderBottom": f"2px solid {ACCENT}"}),
            html.Tbody(table_rows),
        ], style={"width":"100%", "borderCollapse":"collapse"}),
    ])], style={"backgroundColor": CARD_BG, "border": f"1px solid {GRID}", "borderRadius": "10px"})

    # ── 3. Insight principal (pire station) ───────────────────────────────────
    insight_cards = []
    if len(gains_df) > 0:
        worst_g = gains_df.iloc[0]
        if worst_g["Écart (s)"] > 0:
            delta_s = int(worst_g["Écart (s)"])
            # Simuler le gain de rang : si on soustrait delta du total_sec, combien d'athlètes devient-on devant
            simulated_total = row.get("Total_sec", 0) - delta_s
            gain_places = int((d["Total_sec"] < row.get("Total_sec", 0)).sum() -
                               (d["Total_sec"] < simulated_total).sum())
            insight_text = (
                f"Ta plus grande marge : {worst_g['Station']}. "
                f"Tu y passes {sec_to_mmss(worst_g['Ton temps'])} contre "
                f"{sec_to_mmss(worst_g['Médiane catégorie'])} pour la médiane de ta catégorie "
                f"— soit {sec_to_mmss(delta_s)} de perdu. "
                f"En progressant là uniquement, tu gagnerais ~{abs(gain_places)} place(s)."
            )
            insight_cards.append(dbc.Alert([
                html.I(className="bi bi-bullseye me-2"),
                html.Strong("Insight principal : "), insight_text
            ], color="danger", style={"backgroundColor": "#2D0A00", "borderColor": ACCENT, "color": TEXT}))

    # ── 4. Profil Runner vs Worker ────────────────────────────────────────────
    run_total  = row.get("Runs_Total_sec")
    work_total = row.get("Workouts_Total_sec")
    if pd.notna(run_total) and pd.notna(work_total) and len(d_cat) > 1:
        run_pct  = (d_cat["Runs_Total_sec"] < run_total).mean() * 100
        work_pct = (d_cat["Workouts_Total_sec"] < work_total).mean() * 100
        diff = run_pct - work_pct
        if diff > 5:
            profile_text = (f"Tu es un meilleur runner que worker : "
                            f"top {100-run_pct:.0f}% en running, top {100-work_pct:.0f}% en workout "
                            f"dans ta catégorie. Focus sur les stations !")
            profile_color = BLUE
        elif diff < -5:
            profile_text = (f"Tu es un meilleur worker que runner : "
                            f"top {100-run_pct:.0f}% en running, top {100-work_pct:.0f}% en workout "
                            f"dans ta catégorie. Le running est ta priorité !")
            profile_color = GREEN
        else:
            profile_text = (f"Profil équilibré : top {100-run_pct:.0f}% en running, "
                            f"top {100-work_pct:.0f}% en workout dans ta catégorie.")
            profile_color = ORANGE

        insight_cards.append(dbc.Alert([
            html.I(className="bi bi-person-fill-check me-2"),
            html.Strong("Profil Runner vs Worker : "), profile_text
        ], style={"backgroundColor": CARD_BG, "borderColor": profile_color,
                  "color": TEXT, "borderLeft": f"4px solid {profile_color}"}))

    # ── 5. Objectif suivant ───────────────────────────────────────────────────
    if isinstance(rank, (int, float)) and rank > 1:
        target_rank = rank - 1
        target_rows = d[d["Rank"] == target_rank]
        if len(target_rows) > 0:
            target   = target_rows.iloc[0]
            t_name   = target.get("Display_Name", "l'athlète précédent")
            gap_s    = int(row.get("Total_sec", 0) - target.get("Total_sec", 0))
            # Quelle station permettrait de combler l'écart ?
            best_station = None
            best_potential = 0
            for lbl, col in zip(WORKOUT_LABELS, WORKOUT_COLS):
                ath_v = row.get(col)
                tgt_v = target.get(col)
                if pd.notna(ath_v) and pd.notna(tgt_v) and ath_v > tgt_v:
                    potential = ath_v - tgt_v
                    if potential > best_potential:
                        best_potential = potential
                        best_station   = lbl
            station_hint = (f" En améliorant ta station {best_station}, "
                            f"tu pourrais récupérer jusqu'à {sec_to_mmss(int(best_potential))}."
                            if best_station else "")
            target_text = (f"L'athlète juste devant toi est {t_name} "
                           f"(#{target_rank}) — écart de {sec_to_mmss(gap_s)}.{station_hint}")
            insight_cards.append(dbc.Alert([
                html.I(className="bi bi-flag-fill me-2"),
                html.Strong("Objectif suivant : "), target_text
            ], style={"backgroundColor": "#0D1F0D", "borderColor": GREEN,
                      "color": TEXT, "borderLeft": f"4px solid {GREEN}"}))

    return html.Div([
        diag_card,
        gains_table,
        html.Div(style={"height": "12px"}),
        html.Div(insight_cards),
    ])


# ── Lancement ─────────────────────────────────────────────────────────────────
# ── CB7 : Vue macro saison ────────────────────────────────────────────────────
@app.callback(
    Output("season-macro", "children"),
    Input("season-cat-filter", "value"),
)
def update_season_macro(selected_cats):
    events_avail = [e for e in EVENTS_CHRONO if e in df["Event"].unique()]
    if not events_avail:
        return html.Div("Aucun événement disponible.", style={"color": SUBTEXT})

    # Line chart médiane par event par catégorie
    cats = selected_cats if selected_cats else categories_sorted[:3]
    fig_line = go.Figure()
    for cat in cats[:6]:
        medians = []
        for ev in events_avail:
            sub = df[(df["Event"] == ev) & (df["Category"] == cat)]["Total_min"]
            medians.append(sub.median() if len(sub) > 0 else None)
        fig_line.add_trace(go.Scatter(
            x=events_avail, y=medians, mode="lines+markers",
            name=cat.replace("HYROX ", ""),
            connectgaps=True,
            hovertemplate="%{x}<br>Médiane: %{y:.1f} min<extra>" + cat.replace("HYROX ","") + "</extra>",
        ))
    fig_line.update_layout(
        title="Médiane du temps final par événement",
        xaxis_title="Événement", yaxis_title="Temps médian (min)",
        **CHART_LAYOUT
    )

    # Bar chart finishers par event coloré par continent
    finishers = []
    for ev in events_avail:
        sub = df[df["Event"] == ev]
        for cont in sub["Continent"].unique():
            finishers.append({"Event": ev, "Continent": cont,
                               "Count": len(sub[sub["Continent"] == cont])})
    fin_df = pd.DataFrame(finishers)

    cont_colors = {"Europe": BLUE, "North America": GREEN, "Oceania": ORANGE,
                   "South America": "#A855F7", "Africa": "#EC4899", "Asia": "#14B8A6", "Other": SUBTEXT}
    fig_bar = go.Figure()
    if len(fin_df) > 0:
        for cont in fin_df["Continent"].unique():
            sub = fin_df[fin_df["Continent"] == cont]
            fig_bar.add_trace(go.Bar(
                x=sub["Event"], y=sub["Count"], name=cont,
                marker_color=cont_colors.get(cont, SUBTEXT),
                hovertemplate="%{x}<br>" + cont + ": %{y}<extra></extra>",
            ))
    fig_bar.update_layout(
        barmode="stack", title="Nombre de finishers par événement",
        xaxis_title="Événement", yaxis_title="Athlètes",
        **CHART_LAYOUT
    )

    return dbc.Row([
        dbc.Col(card(dcc.Graph(figure=fig_line, config={"displayModeBar": False})), md=7),
        dbc.Col(card(dcc.Graph(figure=fig_bar,  config={"displayModeBar": False})), md=5),
    ], className="g-3")


# ── CB8a : Options dynamiques dropdown athlète saison ─────────────────────────
@app.callback(
    Output("season-athlete-search", "options"),
    Input("season-athlete-search", "search_value"),
    prevent_initial_call=True
)
def update_season_athlete_options(search_value):
    if not search_value or len(search_value.strip()) < 2:
        return []
    q = normalize_str(search_value.strip())
    # Dédupliquer par nom normalisé
    mask = df["Name"].apply(lambda n: q in normalize_str(n))
    matches = df[mask]
    if len(matches) == 0:
        return []
    # Une option par nom unique (Display_Name)
    seen = set()
    options = []
    for _, r in matches.iterrows():
        key = str(r["Name"]).strip().upper()
        if key in seen:
            continue
        seen.add(key)
        n_events = matches[matches["Name"].str.strip().str.upper() == key]["Event"].nunique()
        event_str = f" · {n_events} course(s)" if n_events > 1 else f" · {r.get('Event','')}"
        label = f"{str(r['Name']).strip()}{event_str} ({r.get('Category','')}, {r['Country']})"
        options.append({"label": label, "value": str(r["Name"]).strip()})
        if len(options) >= 50:
            break
    return options


# ── CB8b : Résultat analyse saison ────────────────────────────────────────────
@app.callback(
    Output("season-athlete-result", "children"),
    Input("season-athlete-search", "value"),
    prevent_initial_call=True
)
def search_athlete_season(query):
    if not query:
        return html.Div("Sélectionnez un athlète dans la liste.", style={"color": SUBTEXT})

    # Recherche exacte sur le nom sélectionné
    matches = df[df["Name"].str.strip().str.upper() == str(query).strip().upper()]
    if len(matches) == 0:
        return dbc.Alert(f"Aucun résultat pour « {query} ».", color="warning",
                         style={"backgroundColor": "#2D2000", "color": TEXT, "borderColor": ORANGE})

    # Dédupliquer : un athlète peut être dans plusieurs catégories/events
    # Grouper par nom normalisé pour identifier les individus
    name_groups = matches.groupby(matches["Name"].str.strip().str.upper())
    result_blocks = []

    for norm_name, group in list(name_groups)[:5]:  # max 5 athlètes distincts
        athlete_name = group.iloc[0]["Display_Name"]
        # Trier par ordre chronologique
        group = group.copy()
        group["_ev_order"] = group["Event"].map(
            {e: i for i, e in enumerate(EVENTS_CHRONO)})
        group = group.sort_values("_ev_order")

        events_found = group["Event"].tolist()
        times        = group["Total_sec"].tolist()
        ranks        = group["Rank"].tolist()
        finish_times = group["Finish_Time"].tolist()

        # ── Timeline cards ────────────────────────────────────────────────────
        timeline_items = []
        for i, (_, row) in enumerate(group.iterrows()):
            prev_time = times[i-1] if i > 0 and pd.notna(times[i-1]) else None
            curr_time = row["Total_sec"]
            if i > 0 and prev_time is not None and pd.notna(curr_time):
                delta = curr_time - prev_time
                arrow = f"▼ -{sec_to_mmss(abs(int(delta)))}" if delta < 0 else f"▲ +{sec_to_mmss(abs(int(delta)))}"
                arrow_color = GREEN if delta < 0 else ACCENT
            else:
                arrow, arrow_color = "—", SUBTEXT

            timeline_items.append(
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.Div(str(row["Event"]), style={"color": SUBTEXT, "fontSize": ".7rem",
                                                        "textTransform": "uppercase"}),
                    html.Div(str(row.get("Finish_Time","–")),
                             style={"color": TEXT, "fontWeight": "700", "fontSize": "1.2rem"}),
                    html.Div(f"#{row.get('Rank','?')}", style={"color": SUBTEXT, "fontSize": ".85rem"}),
                    html.Div(arrow, style={"color": arrow_color, "fontWeight": "700",
                                           "fontSize": ".9rem", "marginTop": "4px"}),
                ])], style={"backgroundColor": BG, "border": f"1px solid {GRID}",
                             "textAlign": "center", "borderRadius": "8px"}),
                md=2)
            )

        # ── Line chart progression ─────────────────────────────────────────────
        valid = [(ev, t/60) for ev, t in zip(events_found, times) if pd.notna(t)]
        fig_prog = go.Figure()
        if len(valid) >= 2:
            evs_v, tms_v = zip(*valid)
            x_idx = list(range(len(evs_v)))
            fig_prog.add_trace(go.Scatter(x=list(evs_v), y=list(tms_v), mode="lines+markers",
                                          name=str(athlete_name), line=dict(color=ACCENT, width=3),
                                          marker=dict(size=10)))
            # Tendance linéaire
            coeffs = np.polyfit(x_idx, list(tms_v), 1)
            trend  = np.polyval(coeffs, x_idx)
            trend_color = GREEN if coeffs[0] < 0 else "#EF4444"
            fig_prog.add_trace(go.Scatter(x=list(evs_v), y=list(trend), mode="lines",
                                          name="Tendance", line=dict(color=trend_color,
                                                                      width=2, dash="dash")))
        elif len(valid) == 1:
            fig_prog.add_trace(go.Scatter(x=[valid[0][0]], y=[valid[0][1]], mode="markers",
                                          marker=dict(color=ACCENT, size=12)))
        fig_prog.update_layout(title=f"Progression — {athlete_name}",
                               yaxis_title="Temps (min)", **CHART_LAYOUT)

        # ── Radar multi-events par station ─────────────────────────────────────
        fig_radar = go.Figure()
        palette = [ACCENT, BLUE, GREEN, ORANGE, "#A855F7", "#EC4899"]
        for i2, (_, row) in enumerate(group.iterrows()):
            vals = [row.get(c, None) for c in WORKOUT_COLS]
            vals_min = [v/60 if pd.notna(v) else 0 for v in vals]
            if any(v > 0 for v in vals_min):
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_min + [vals_min[0]],
                    theta=WORKOUT_LABELS + [WORKOUT_LABELS[0]],
                    fill="toself", name=str(row["Event"]),
                    line=dict(color=palette[i2 % len(palette)]),
                    fillcolor=f"rgba({int(palette[i2%len(palette)][1:3],16)},"
                              f"{int(palette[i2%len(palette)][3:5],16)},"
                              f"{int(palette[i2%len(palette)][5:7],16)},0.1)",
                ))
        fig_radar.update_layout(
            title="Stations par événement",
            polar=dict(radialaxis=dict(visible=True, gridcolor=GRID, color=SUBTEXT),
                       bgcolor=PLOTBG, angularaxis=dict(color=SUBTEXT)),
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ["xaxis","yaxis"]},
        )

        # ── Tableau récap ──────────────────────────────────────────────────────
        recap_rows = []
        prev_time, prev_rank = None, None
        for _, row in group.iterrows():
            t = row.get("Total_sec")
            r = row.get("Rank")
            dt = sec_to_mmss(abs(int(t - prev_time))) if pd.notna(t) and prev_time is not None else "–"
            dt_sign = ("▼ -" if t < prev_time else "▲ +") if pd.notna(t) and prev_time is not None else ""
            dt_color = (GREEN if pd.notna(t) and prev_time and t < prev_time
                        else ACCENT if pd.notna(t) and prev_time else SUBTEXT)
            dr = (int(r) - int(prev_rank)) if pd.notna(r) and prev_rank is not None else None
            dr_str = (f"▼ {abs(dr)}" if dr and dr < 0 else f"▲ +{dr}" if dr else "–")
            dr_color = (GREEN if dr and dr < 0 else ACCENT if dr and dr > 0 else SUBTEXT)
            recap_rows.append(html.Tr([
                html.Td(str(row["Event"]),      style={"color": TEXT,      "padding": "6px 10px"}),
                html.Td(str(row.get("Category","–")), style={"color": SUBTEXT, "padding": "6px 10px"}),
                html.Td(str(row.get("Finish_Time","–")), style={"color": ACCENT, "padding": "6px 10px",
                                                                  "fontWeight":"700"}),
                html.Td(f"#{r}",               style={"color": TEXT,      "padding": "6px 10px"}),
                html.Td(f"#{row.get('Rank_AG','?')}", style={"color": SUBTEXT, "padding": "6px 10px"}),
                html.Td(f"{dt_sign}{dt}",      style={"color": dt_color,  "padding": "6px 10px",
                                                        "fontWeight":"700"}),
                html.Td(dr_str,                style={"color": dr_color,  "padding": "6px 10px",
                                                       "fontWeight":"700"}),
            ], style={"borderBottom": f"1px solid {GRID}"}))
            prev_time, prev_rank = t, r

        recap_table = html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"color": SUBTEXT, "padding": "6px 10px", "textAlign": "left"})
                for h in ["Événement","Catégorie","Temps","Rang","Rang AG","Δ Temps","Δ Rang"]
            ]), style={"borderBottom": f"2px solid {ACCENT}"}),
            html.Tbody(recap_rows),
        ], style={"width": "100%", "borderCollapse": "collapse"})

        result_blocks.append(dbc.Card([dbc.CardBody([
            html.H5(str(athlete_name), style={"color": TEXT, "fontWeight": "800"}),
            html.Div(f"{len(group)} apparition(s) trouvée(s)",
                     style={"color": SUBTEXT, "fontSize": ".8rem", "marginBottom": "12px"}),
            dbc.Row(timeline_items, className="g-2 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_prog,  config={"displayModeBar": False})), md=7),
                dbc.Col(card(dcc.Graph(figure=fig_radar, config={"displayModeBar": False})), md=5),
            ], className="g-3 mb-3"),
            card(recap_table),
        ])], style={"backgroundColor": CARD_BG, "border": f"1px solid {GRID}",
                    "borderRadius": "10px", "marginBottom": "16px"}))

    return html.Div(result_blocks)


# ── CB9 : Benchmark inter-events ──────────────────────────────────────────────
@app.callback(
    Output("season-benchmark", "children"),
    [Input("bench-event-a", "value"),
     Input("bench-event-b", "value"),
     Input("bench-category", "value")],
)
def update_benchmark(ev_a, ev_b, cat):
    if not ev_a or not ev_b or not cat:
        return html.Div("Sélectionnez deux événements et une catégorie.", style={"color": SUBTEXT})
    if ev_a == ev_b:
        return dbc.Alert("Choisissez deux événements différents.", color="warning",
                         style={"backgroundColor": "#2D2000", "color": TEXT, "borderColor": ORANGE})

    da = df[(df["Event"] == ev_a) & (df["Category"] == cat)]
    db = df[(df["Event"] == ev_b) & (df["Category"] == cat)]
    if len(da) == 0 or len(db) == 0:
        return dbc.Alert(f"Catégorie « {cat} » absente dans l'un des événements.",
                         color="warning",
                         style={"backgroundColor": "#2D2000", "color": TEXT, "borderColor": ORANGE})

    # Violin plot
    fig_violin = go.Figure()
    fig_violin.add_trace(go.Violin(y=da["Total_min"].dropna(), name=ev_a,
                                    box_visible=True, meanline_visible=True,
                                    line_color=ACCENT, fillcolor="rgba(232,71,43,0.2)"))
    fig_violin.add_trace(go.Violin(y=db["Total_min"].dropna(), name=ev_b,
                                    box_visible=True, meanline_visible=True,
                                    line_color=BLUE, fillcolor="rgba(59,130,246,0.2)"))
    fig_violin.update_layout(title=f"Distribution des temps — {cat}",
                              yaxis_title="Temps (min)", **CHART_LAYOUT)

    # Bar chart stations
    station_avgs_a = [da[c].mean() for c in WORKOUT_COLS if c in da.columns]
    station_avgs_b = [db[c].mean() for c in WORKOUT_COLS if c in db.columns]
    labels_avail   = WORKOUT_LABELS[:min(len(station_avgs_a), len(station_avgs_b))]
    fig_stations = go.Figure()
    fig_stations.add_trace(go.Bar(name=ev_a, x=labels_avail,
                                   y=[v/60 for v in station_avgs_a[:len(labels_avail)]],
                                   marker_color=ACCENT,
                                   text=[sec_to_mmss(v) for v in station_avgs_a[:len(labels_avail)]],
                                   textposition="outside"))
    fig_stations.add_trace(go.Bar(name=ev_b, x=labels_avail,
                                   y=[v/60 for v in station_avgs_b[:len(labels_avail)]],
                                   marker_color=BLUE,
                                   text=[sec_to_mmss(v) for v in station_avgs_b[:len(labels_avail)]],
                                   textposition="outside"))
    fig_stations.update_layout(barmode="group",
                                title="Temps moyen par station",
                                yaxis_title="Temps (min)", **CHART_LAYOUT)

    # Phrase auto-générée
    med_a = da["Total_min"].median()
    med_b = db["Total_min"].median()
    delta_total = abs(med_a - med_b)
    faster_ev   = ev_a if med_a < med_b else ev_b
    delta_mmss  = sec_to_mmss(int(delta_total * 60))

    # Station avec la plus grande différence
    station_deltas = []
    for lbl, col in zip(WORKOUT_LABELS, WORKOUT_COLS):
        if col in da.columns and col in db.columns:
            d_delta = abs(da[col].mean() - db[col].mean())
            station_deltas.append((lbl, d_delta))
    biggest_station, biggest_delta = max(station_deltas, key=lambda x: x[1]) if station_deltas else ("–", 0)

    insight_text = (
        f"À {ev_a}, la médiane {cat.replace('HYROX ','')} est {sec_to_mmss(int(med_a*60))}, "
        f"contre {sec_to_mmss(int(med_b*60))} à {ev_b}. "
        f"{faster_ev} est plus rapide de {delta_mmss}. "
        f"La plus grande différence par station vient du {biggest_station} "
        f"({sec_to_mmss(int(biggest_delta))} en moyenne)."
    )

    return html.Div([
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig_violin,   config={"displayModeBar": False})), md=5),
            dbc.Col(card(dcc.Graph(figure=fig_stations, config={"displayModeBar": False})), md=7),
        ], className="g-3 mb-3"),
        dbc.Alert([html.I(className="bi bi-chat-quote-fill me-2"), insight_text],
                  style={"backgroundColor": "#0D1F2D", "borderColor": BLUE,
                         "color": TEXT, "borderLeft": f"4px solid {BLUE}",
                         "fontSize": "1rem"}),
    ])


if __name__ == "__main__":
    print("\n✅  Dashboard Hyrox Season 8 – All Events 2025-2026")
    print("🌐  Ouvre http://127.0.0.1:8050 dans ton navigateur\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
