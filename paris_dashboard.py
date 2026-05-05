"""
Hyrox Season 8 – Paris 2026 – Dashboard
Run: python3 paris_dashboard.py → http://127.0.0.1:8051
"""

import os, unicodedata
from functools import lru_cache
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

def normalize_str(s):
    return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower()

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "paris2026_data.parquet")
pd.options.mode.chained_assignment = None

WORKOUT_COLS   = ["SkiErg_sec","SledPush_sec","SledPull_sec","BurpeeBJ_sec",
                  "Row_sec","FarmersCarry_sec","SandbagLunges_sec","WallBalls_sec"]
WORKOUT_LABELS = ["Ski Erg","Sled Push","Sled Pull","Burpee BJ",
                  "Row","Farmers Carry","Sandbag Lunges","Wall Balls"]
RUN_COLS    = [f"Run{i}_sec" for i in range(1, 9)]
RUN_LABELS  = [f"Run {i}" for i in range(1, 9)]
SCORE_COLS  = ["SkiErg_Score","SledPush_Score","SledPull_Score","BurpeeBJ_Score",
               "Row_Score","FarmersCarry_Score","SandbagLunges_Score","WallBalls_Score"]

print("Chargement des données Paris 2026...", flush=True)
_raw = pd.read_parquet(DATA_PATH)
for _c in ["Country","Age_Group","Category","Finish_Time","Name","Team_Name"]:
    if _c in _raw.columns:
        _raw[_c] = _raw[_c].astype(str).replace("nan","")
df = _raw.reset_index(drop=True)
del _raw

df["Total_min"]    = df["Total_sec"] / 60
df["Runs_min"]     = df["Runs_Total_sec"] / 60
df["Workouts_min"] = df["Workouts_Total_sec"] / 60
df["Roxzone_min"]  = df["Roxzone_sec"] / 60
df["Display_Name"] = df.apply(
    lambda r: r["Team_Name"] if r.get("Team_Name","") not in ("","nan") else r["Name"], axis=1)
df["Country"]   = df["Country"].replace("","").fillna("")
df["Age_Group"] = df["Age_Group"].replace("","–").fillna("–")

_total_rw = df["Runs_Total_sec"] + df["Workouts_Total_sec"]
df["Balance_Score"] = (df["Runs_Total_sec"] / _total_rw.replace(0, np.nan) * 100).round(1)
del _total_rw

TOTAL = len(df)
N_COUNTRIES = df["Country"].nunique()
N_CATS = df["Category"].nunique()

categories_sorted  = sorted(df["Category"].dropna().unique())
countries_sorted   = sorted(c for c in df["Country"].dropna().unique() if c)
age_groups_sorted  = sorted(df["Age_Group"].dropna().unique())
print(f"OK : {TOTAL} lignes · {N_CATS} catégories · {N_COUNTRIES} pays", flush=True)

# ── Thème ─────────────────────────────────────────────────────────────────────
ACCENT  = "#E8472B"; BG = "#0F1117"; CARD_BG = "#1A1D27"
TEXT    = "#F0F2F8"; SUBTEXT = "#9CA3AF"; PLOTBG = "#13151F"; GRID = "#2A2D3A"
COLORS  = px.colors.qualitative.Vivid
GREEN   = "#10B981"; BLUE = "#3B82F6"; ORANGE = "#F59E0B"

CHART_LAYOUT = dict(
    paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT),
)

_dd_style = {"backgroundColor": CARD_BG, "color": TEXT, "border": f"1px solid {GRID}"}

def sec_to_mmss(s):
    if pd.isna(s): return "N/A"
    s = int(s)
    if s >= 3600:
        h = s//3600; m = (s%3600)//60; sec = s%60
        return f"{h}h{m:02d}:{sec:02d}"
    return f"{s//60}:{s%60:02d}"

def kpi_card(icon, title, value, sub=""):
    return dbc.Card([dbc.CardBody([
        html.Div([
            html.I(className=f"bi {icon} fs-3", style={"color": ACCENT}),
            html.Div([
                html.Div(value, style={"fontSize":"1.6rem","fontWeight":"700","color":TEXT}),
                html.Div(title,  style={"fontSize":".75rem","color":SUBTEXT,
                                        "textTransform":"uppercase","letterSpacing":".08em"}),
                html.Div(sub,    style={"fontSize":".7rem","color":SUBTEXT}) if sub else None,
            ], style={"marginLeft":"12px"}),
        ], style={"display":"flex","alignItems":"center"}),
    ])], style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px"})

def card(children, **kw):
    return dbc.Card(dbc.CardBody(children),
                    style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}",
                           "borderRadius":"10px",**kw})

def sample_df(d, n=5000):
    return d.sample(n, random_state=42) if len(d) > n else d

def build_athlete_options(d):
    if "Team_Name" in d.columns and "Athlete_Position" in d.columns:
        solo = d[d["Team_Name"].isna() | (d["Team_Name"] == "")]
        team = d[d["Team_Name"].notna() & (d["Team_Name"] != "")]
        team = team[team["Athlete_Position"] == 1]
        d_dedup = pd.concat([solo, team]).sort_values("Rank")
    else:
        d_dedup = d
    options = []
    for idx, r in d_dedup.iterrows():
        country = r["Country"] if r["Country"] else ""
        label = (f"#{r['Rank']} {r['Display_Name']} "
                 f"({r.get('Category','')}"
                 f"{', '+country if country else ''}"
                 f", {r['Age_Group']})")
        options.append({"label": label, "value": idx})
    return options

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__,
           external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
           suppress_callback_exceptions=True,
           title="Hyrox Paris 2026 – Dashboard")
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

app.layout = html.Div(
    style={"backgroundColor":BG,"minHeight":"100vh","fontFamily":"Inter, sans-serif"},
    children=[
        dcc.Store(id="store", data={"cats":[],"countries":[],"ags":[],"topn":TOTAL}),

        # Header
        html.Div([
            html.Div([
                html.Span("⚡", style={"fontSize":"1.6rem","marginRight":"10px"}),
                html.Span("HYROX Paris 2026", style={"fontSize":"1.5rem","fontWeight":"800","color":TEXT}),
                html.Span(" · Dashboard Analytique", style={"fontSize":"1.1rem","color":SUBTEXT}),
            ], style={"display":"flex","alignItems":"center"}),
            html.Div(
                f"{TOTAL:,} athlètes · {N_CATS} catégories · {N_COUNTRIES} pays".replace(",","·"),
                style={"color":SUBTEXT,"fontSize":".8rem"}),
        ], style={"backgroundColor":CARD_BG,"padding":"16px 32px",
                  "borderBottom":f"2px solid {ACCENT}",
                  "display":"flex","justifyContent":"space-between","alignItems":"center"}),

        html.Div(style={"padding":"24px 32px"}, children=[

            # Filtres
            dbc.Row([
                dbc.Col([
                    html.Label("Catégorie", style={"color":SUBTEXT,"fontSize":".75rem"}),
                    dcc.Dropdown(id="f-cat",
                                 options=[{"label":c,"value":c} for c in categories_sorted],
                                 multi=True, placeholder="Toutes", style=_dd_style),
                ], md=4),
                dbc.Col([
                    html.Label("Pays", style={"color":SUBTEXT,"fontSize":".75rem"}),
                    dcc.Dropdown(id="f-country",
                                 options=[{"label":c,"value":c} for c in countries_sorted],
                                 multi=True, placeholder="Tous", style=_dd_style),
                ], md=3),
                dbc.Col([
                    html.Label("Tranche d'âge", style={"color":SUBTEXT,"fontSize":".75rem"}),
                    dcc.Dropdown(id="f-ag",
                                 options=[{"label":a,"value":a} for a in age_groups_sorted],
                                 multi=True, placeholder="Toutes", style=_dd_style),
                ], md=3),
                dbc.Col([
                    html.Label("Top N", style={"color":SUBTEXT,"fontSize":".75rem"}),
                    dcc.Slider(id="f-topn", min=10, max=TOTAL, step=50, value=TOTAL,
                               marks={10:"10",500:"500",1000:"1k",5000:"5k",TOTAL:"All"},
                               tooltip={"placement":"bottom"}, updatemode="mouseup",
                               className="mt-1"),
                ], md=2),
            ], className="mb-4 p-3",
               style={"backgroundColor":CARD_BG,"borderRadius":"10px","border":f"1px solid {GRID}"}),

            # KPIs
            html.Div(id="kpis", className="mb-4"),

            # Tabs
            dcc.Tabs(id="tabs", value="overview", children=[
                dcc.Tab(label="🏁 Vue d'ensemble",   value="overview",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="💪 Stations Workout", value="workouts",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="🏃 Running",          value="running",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="🌍 Géographie",       value="geo",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="👥 Démographie",      value="demo",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="🔍 Profil Athlète",   value="athlete",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="🧠 Analyse Coach",    value="coach",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="⚔️ Face-à-Face",      value="faceaface",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="📋 Classement",       value="ranking",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
                dcc.Tab(label="🔬 Insights Exclusifs", value="insights",
                        style={"backgroundColor":BG,"color":SUBTEXT},
                        selected_style={"backgroundColor":CARD_BG,"color":TEXT,"borderTop":f"2px solid {ACCENT}"}),
            ], style={"marginBottom":"20px"}),

            dcc.Loading(type="dot", color=ACCENT,
                        children=html.Div(id="tab-content")),
        ]),
    ]
)

# ── Cache filtre ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def _filt(cats, countries, ags, topn):
    mask = pd.Series(True, index=df.index)
    if cats:      mask &= df["Category"].isin(cats)
    if countries: mask &= df["Country"].isin(countries)
    if ags:       mask &= df["Age_Group"].isin(ags)
    idx = df.index[mask]
    if topn and topn < len(idx): idx = idx[:topn]
    return idx.tolist()

def get_d(store):
    if store is None: return df
    cat = tuple(sorted(store.get("cats",[]) or []))
    co  = tuple(sorted(store.get("countries",[]) or []))
    ag  = tuple(sorted(store.get("ags",[]) or []))
    tn  = store.get("topn", TOTAL) or TOTAL
    return df.loc[_filt(cat, co, ag, tn)]

# ── CB1 : Store ───────────────────────────────────────────────────────────────
@app.callback(Output("store","data"),
              [Input("f-cat","value"), Input("f-country","value"),
               Input("f-ag","value"),  Input("f-topn","value")])
def upd_store(cats, countries, ags, topn):
    return {"cats":cats or [],"countries":countries or [],
            "ags":ags or [],"topn":topn or TOTAL}

# ── CB2 : KPIs ────────────────────────────────────────────────────────────────
@app.callback(Output("kpis","children"), Input("store","data"))
def upd_kpis(store):
    d = get_d(store)
    if len(d) == 0:
        return dbc.Row([dbc.Col(kpi_card("bi-people-fill","Athlètes","0"),md=3)])
    best_idx = d["Total_sec"].idxmin()
    best  = sec_to_mmss(d.at[best_idx,"Total_sec"])
    bname = d.at[best_idx,"Display_Name"]
    avg = sec_to_mmss(d["Total_sec"].mean())
    med = sec_to_mmss(d["Total_sec"].median())
    nc  = d["Country"].nunique()
    return dbc.Row([
        dbc.Col(kpi_card("bi-people-fill",   "Athlètes",      str(len(d)), f"{nc} pays"), md=3),
        dbc.Col(kpi_card("bi-trophy-fill",   "Meilleur temps", best,       bname),        md=3),
        dbc.Col(kpi_card("bi-bar-chart-fill","Temps moyen",    avg,        ""),           md=3),
        dbc.Col(kpi_card("bi-activity",      "Temps médian",   med,        ""),           md=3),
    ], className="g-3")

# ── CB3 : Tabs ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content","children"),
              [Input("tabs","value"), Input("store","data")])
def render_tab(tab, store):
    d = get_d(store)
    if len(d) == 0:
        return html.Div("Aucune donnée.", style={"color":SUBTEXT,"padding":"40px"})

    # ── Vue d'ensemble ────────────────────────────────────────────────────────
    if tab == "overview":
        fig_hist = go.Figure(go.Histogram(x=d["Total_min"], nbinsx=60,
                                          marker_color=ACCENT, opacity=0.85))
        med = d["Total_min"].median()
        fig_hist.add_vline(x=med, line_dash="dash", line_color=BLUE,
                           annotation_text=f"Médiane {sec_to_mmss(med*60)}",
                           annotation_font_color=BLUE)
        fig_hist.update_layout(title="Distribution des temps finaux",
                               xaxis_title="Temps (min)", yaxis_title="Athlètes", **CHART_LAYOUT)

        fig_donut = go.Figure(go.Pie(
            labels=["Running","Workout","Roxzone"],
            values=[d["Runs_min"].mean(), d["Workouts_min"].mean(), d["Roxzone_min"].mean()],
            hole=0.6, marker_colors=[ACCENT, BLUE, GREEN], textinfo="label+percent"))
        fig_donut.update_layout(title="Décomposition moyenne", **CHART_LAYOUT)

        top20 = d.head(20).copy()
        top20["label"] = top20["Display_Name"].str.split(",").str[0].str.split().str[-1] \
                         + " (" + top20["Country"] + ")"
        fig_top = go.Figure(go.Bar(
            y=top20["label"][::-1], x=top20["Total_min"][::-1], orientation="h",
            marker_color=ACCENT, text=top20["Finish_Time"][::-1], textposition="outside"))
        fig_top.update_layout(title="Top 20 athlètes", xaxis_title="Temps (min)",
                              height=550, **CHART_LAYOUT)

        ds = sample_df(d)
        fig_scat = px.scatter(ds, x="Runs_min", y="Workouts_min", color="Age_Group",
                              hover_name="Display_Name",
                              labels={"Runs_min":"Running (min)","Workouts_min":"Workout (min)"},
                              title=f"Running vs Workout ({len(ds):,} pts)".replace(",","·"),
                              color_discrete_sequence=COLORS)
        fig_scat.update_layout(**CHART_LAYOUT)
        fig_scat.update_traces(marker=dict(size=4, opacity=0.6))

        # Balance Score
        bal = d["Balance_Score"].dropna()
        avg_bal = bal.mean()
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Histogram(x=bal, nbinsx=40, marker_color=BLUE, opacity=0.8))
        fig_bal.add_vline(x=avg_bal, line_dash="dash", line_color=ACCENT,
                          annotation_text=f"Moy {avg_bal:.1f}%", annotation_font_color=ACCENT)
        fig_bal.add_vline(x=50, line_dash="dot", line_color=GREEN,
                          annotation_text="Équilibre 50%", annotation_font_color=GREEN)
        fig_bal.update_layout(title="Score équilibre Running/Workout (% en running)",
                              xaxis_title="% running", yaxis_title="Athlètes", **CHART_LAYOUT)

        n_runners  = int((bal > 55).sum())
        n_balanced = int(((bal >= 45) & (bal <= 55)).sum())
        n_workers  = int((bal < 45).sum())
        fig_prof = go.Figure(go.Pie(
            labels=["Runners (>55%)","Équilibrés (45-55%)","Workers (<45%)"],
            values=[n_runners, n_balanced, n_workers],
            hole=0.55, marker_colors=[BLUE, GREEN, ACCENT], textinfo="label+percent"))
        fig_prof.update_layout(title="Répartition des profils", **CHART_LAYOUT)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_hist,  config={"displayModeBar":False})), md=8),
                dbc.Col(card(dcc.Graph(figure=fig_donut, config={"displayModeBar":False})), md=4),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_top,   config={"displayModeBar":False})), md=5),
                dbc.Col(card(dcc.Graph(figure=fig_scat,  config={"displayModeBar":False})), md=7),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_bal,   config={"displayModeBar":False})), md=8),
                dbc.Col(card(dcc.Graph(figure=fig_prof,  config={"displayModeBar":False})), md=4),
            ], className="g-3"),
        ])

    # ── Stations Workout ──────────────────────────────────────────────────────
    elif tab == "workouts":
        avgs = [d[c].mean() for c in WORKOUT_COLS if c in d.columns]
        labs = WORKOUT_LABELS[:len(avgs)]
        fig_bar = go.Figure(go.Bar(x=labs, y=[a/60 for a in avgs], marker_color=ACCENT,
                                   text=[sec_to_mmss(a) for a in avgs], textposition="outside"))
        fig_bar.update_layout(title="Temps moyen par station", yaxis_title="Temps (min)", **CHART_LAYOUT)

        fig_box = go.Figure()
        for col, label in zip(WORKOUT_COLS, WORKOUT_LABELS):
            if col in d.columns:
                fig_box.add_trace(go.Box(y=d[col].dropna()/60, name=label,
                                         marker_color=ACCENT, boxmean=True))
        fig_box.update_layout(title="Distribution par station", yaxis_title="Temps (min)",
                               showlegend=False, **CHART_LAYOUT)

        cols_c = [c for c in WORKOUT_COLS if c in d.columns]
        corr = d[cols_c].corr()
        corr.index = corr.columns = WORKOUT_LABELS[:len(cols_c)]
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Corrélations entre stations")
        fig_corr.update_layout(**CHART_LAYOUT)

        score_avgs = [d[c].mean() for c in SCORE_COLS if c in d.columns]
        fig_score = go.Figure(go.Bar(
            x=WORKOUT_LABELS[:len(score_avgs)], y=score_avgs,
            marker=dict(color=score_avgs, colorscale=[[0,"#EF4444"],[0.5,ORANGE],[1,GREEN]],
                        showscale=True, cmin=0, cmax=100),
            text=[f"{s:.1f}" for s in score_avgs], textposition="outside"))
        sl = {**CHART_LAYOUT, "yaxis":{**CHART_LAYOUT.get("yaxis",{}),"range":[0,105]}}
        fig_score.update_layout(title="Score moyen par station", **sl)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_bar,   config={"displayModeBar":False})), md=6),
                dbc.Col(card(dcc.Graph(figure=fig_score, config={"displayModeBar":False})), md=6),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_box,  config={"displayModeBar":False})), md=8),
                dbc.Col(card(dcc.Graph(figure=fig_corr, config={"displayModeBar":False})), md=4),
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
        med_r    = [d[c].median()/60    for c in RUN_COLS if c in d.columns]
        top_r    = [top10[c].mean()/60  for c in RUN_COLS if c in d.columns]
        bot_r    = [d.tail(100)[c].mean()/60 for c in RUN_COLS if c in d.columns]
        fig_lines = go.Figure()
        fig_lines.add_trace(go.Scatter(x=RUN_LABELS[:len(top_r)], y=top_r, mode="lines+markers",
                                       name="Top 10", line=dict(color=ACCENT,width=3)))
        fig_lines.add_trace(go.Scatter(x=RUN_LABELS[:len(med_r)], y=med_r, mode="lines+markers",
                                       name="Médiane", line=dict(color=BLUE,width=2)))
        fig_lines.add_trace(go.Scatter(x=RUN_LABELS[:len(bot_r)], y=bot_r, mode="lines+markers",
                                       name="Bas 100", line=dict(color=SUBTEXT,width=2,dash="dot")))
        fig_lines.update_layout(title="Top 10 / Médiane / Bas 100",
                                yaxis_title="Temps (min)", **CHART_LAYOUT)

        # Dégradation du rythme (run 1 vs run 8)
        run_med = [d[c].median()/60 for c in RUN_COLS if c in d.columns]
        if len(run_med) >= 2:
            degradation = (run_med[-1] - run_med[0]) / run_med[0] * 100
            deg_text = f"Dégradation Run 1→8 : +{degradation:.1f}% pour la médiane"
        else:
            deg_text = ""

        insight = dbc.Alert([html.I(className="bi bi-graph-up me-2"), deg_text],
                            style={"backgroundColor":"#0D1F2D","borderColor":BLUE,
                                   "color":TEXT,"borderLeft":f"4px solid {BLUE}"}) if deg_text else html.Div()

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_run,   config={"displayModeBar":False})), md=5),
                dbc.Col(card(dcc.Graph(figure=fig_lines, config={"displayModeBar":False})), md=7),
            ], className="g-3 mb-3"),
            insight,
        ])

    # ── Géographie ────────────────────────────────────────────────────────────
    elif tab == "geo":
        d_geo = d[d["Country"].astype(str).str.strip() != ""].copy()
        by_country = (d_geo.groupby("Country", sort=False, observed=True)
                       .agg(Athletes=("Name","count"),
                            Avg_min=("Total_min","mean"),
                            Best_min=("Total_min","min"),
                            Median_min=("Total_min","median"))
                       .round(2).reset_index()
                       .sort_values("Athletes", ascending=False))

        fig_map = px.choropleth(by_country, locations="Country", locationmode="ISO-3",
                                color="Athletes", hover_name="Country",
                                hover_data={"Athletes":True,"Avg_min":":.1f","Best_min":":.1f"},
                                color_continuous_scale="Reds", title="Carte mondiale des participants")
        fig_map.update_layout(
            geo=dict(bgcolor=PLOTBG, lakecolor=PLOTBG, landcolor=CARD_BG,
                     showframe=False, showcoastlines=True, coastlinecolor=GRID,
                     showland=True, showcountries=True, countrycolor=GRID),
            coloraxis_colorbar=dict(title="Athlètes", tickfont=dict(color=TEXT),
                                    titlefont=dict(color=TEXT)),
            **{k:v for k,v in CHART_LAYOUT.items() if k not in ["xaxis","yaxis"]}, height=420)

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
                              labels={"Avg_min":"Temps moyen (min)"},
                              title="Volume vs Performance")
        fig_scat.update_layout(**CHART_LAYOUT)
        fig_scat.update_traces(textposition="top center", marker=dict(sizemin=6))

        top5 = by_country.head(5)["Country"].tolist()
        d5   = d_geo[d_geo["Country"].isin(top5)]
        fig_vio = go.Figure()
        for c in top5:
            fig_vio.add_trace(go.Violin(y=d5[d5["Country"]==c]["Total_min"].dropna(),
                                        name=c, box_visible=True, meanline_visible=True,
                                        line_color=ACCENT))
        fig_vio.update_layout(title="Distribution – Top 5 pays",
                              yaxis_title="Temps (min)", **CHART_LAYOUT)

        return html.Div([
            dbc.Row([dbc.Col(card(dcc.Graph(figure=fig_map, config={"displayModeBar":False})), md=12)],
                    className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_pays, config={"displayModeBar":False})), md=5),
                dbc.Col(card(dcc.Graph(figure=fig_scat, config={"displayModeBar":False})), md=7),
            ], className="g-3 mb-3"),
            dbc.Row([dbc.Col(card(dcc.Graph(figure=fig_vio, config={"displayModeBar":False})), md=12)],
                    className="g-3"),
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

        by_cat = (d.groupby("Category", sort=False, observed=True)
                   .agg(Athletes=("Name","count"),
                        Median_min=("Total_min","median"),
                        Best_min=("Total_min","min"))
                   .round(2).reset_index()
                   .sort_values("Athletes", ascending=False))

        fig_cat = go.Figure(go.Bar(
            x=by_cat["Category"].str.replace("HYROX ",""),
            y=by_cat["Athletes"], marker_color=ACCENT,
            text=by_cat["Athletes"], textposition="outside"))
        fig_cat.update_layout(title="Athlètes par catégorie",
                              xaxis_tickangle=-30, **CHART_LAYOUT)

        fig_ag = go.Figure()
        fig_ag.add_trace(go.Scatter(x=by_ag["Age_Group"], y=by_ag["Avg_min"],
                                    name="Moyen", mode="lines+markers",
                                    line=dict(color=ACCENT,width=3)))
        fig_ag.add_trace(go.Scatter(x=by_ag["Age_Group"], y=by_ag["Median_min"],
                                    name="Médian", mode="lines+markers",
                                    line=dict(color=BLUE,width=2)))
        fig_ag.add_trace(go.Scatter(x=by_ag["Age_Group"], y=by_ag["Best_min"],
                                    name="Meilleur", mode="lines+markers",
                                    line=dict(color=GREEN,width=2,dash="dash")))
        fig_ag.update_layout(title="Temps par tranche d'âge",
                             yaxis_title="Temps (min)", **CHART_LAYOUT)

        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(name="Running", x=by_ag["Age_Group"],
                                   y=by_ag["Avg_Run_min"], marker_color=BLUE))
        fig_stack.add_trace(go.Bar(name="Workout", x=by_ag["Age_Group"],
                                   y=by_ag["Avg_Work_min"], marker_color=ACCENT))
        fig_stack.update_layout(barmode="stack", title="Running/Workout par âge",
                                yaxis_title="Temps (min)", **CHART_LAYOUT)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_cat,   config={"displayModeBar":False})), md=5),
                dbc.Col(card(dcc.Graph(figure=fig_ag,    config={"displayModeBar":False})), md=7),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_stack, config={"displayModeBar":False})), md=12),
            ], className="g-3"),
        ])

    # ── Profil Athlète ────────────────────────────────────────────────────────
    elif tab == "athlete":
        opts = build_athlete_options(d)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label(f"Sélectionner un athlète ({len(opts):,} athlètes)".replace(",","·"),
                               style={"color":SUBTEXT}),
                    dcc.Dropdown(id="ath-sel", options=opts,
                                 value=opts[0]["value"] if opts else None,
                                 style=_dd_style),
                ], md=6),
                dbc.Col([
                    html.Label("Comparer avec", style={"color":SUBTEXT}),
                    dcc.Dropdown(id="ath-cmp",
                                 options=[{"label":"Médiane","value":"median"},
                                          {"label":"Top 10%","value":"top10"}],
                                 value="median", style=_dd_style),
                ], md=3),
            ], className="mb-3"),
            html.Div(id="ath-content"),
        ])

    # ── Analyse Coach ─────────────────────────────────────────────────────────
    elif tab == "coach":
        opts = build_athlete_options(d)
        return html.Div([
            dbc.Row([dbc.Col([
                html.Label(f"Sélectionner un athlète ({len(opts):,} athlètes)".replace(",","·"),
                           style={"color":SUBTEXT}),
                dcc.Dropdown(id="coach-sel", options=opts,
                             value=opts[0]["value"] if opts else None, style=_dd_style),
            ], md=7)], className="mb-3"),
            html.Div(id="coach-content"),
        ])

    # ── Face-à-Face ───────────────────────────────────────────────────────────
    elif tab == "faceaface":
        opts = build_athlete_options(d)
        return html.Div([
            dbc.Row([
                dbc.Col([html.Label("Athlète A", style={"color":SUBTEXT}),
                         dcc.Dropdown(id="ff-a", options=opts,
                                      value=opts[0]["value"] if opts else None, style=_dd_style)], md=5),
                dbc.Col(html.Div("VS", style={"color":ACCENT,"fontSize":"2rem","fontWeight":"800",
                                              "textAlign":"center","marginTop":"20px"}), md=2),
                dbc.Col([html.Label("Athlète B", style={"color":SUBTEXT}),
                         dcc.Dropdown(id="ff-b", options=opts,
                                      value=opts[1]["value"] if len(opts)>1 else None, style=_dd_style)], md=5),
            ], className="mb-4"),
            dcc.Loading(type="dot", color=ACCENT, children=html.Div(id="ff-content")),
        ])

    # ── Classement ────────────────────────────────────────────────────────────
    elif tab == "ranking":
        cols = ["Rank","Rank_AG","Category","Display_Name","Country","Age_Group","Finish_Time",
                "Runs_Total_sec","Workouts_Total_sec","Roxzone_sec"]
        dt = d[[c for c in cols if c in d.columns]].head(1000).copy()
        dt["Running"] = dt["Runs_Total_sec"].apply(sec_to_mmss)
        dt["Workout"] = dt["Workouts_Total_sec"].apply(sec_to_mmss)
        dt["Roxzone"] = dt["Roxzone_sec"].apply(sec_to_mmss)
        dt.drop(columns=["Runs_Total_sec","Workouts_Total_sec","Roxzone_sec"],
                inplace=True, errors="ignore")
        dt.rename(columns={"Display_Name":"Nom(s)"}, inplace=True)
        return card(dash_table.DataTable(
            data=dt.to_dict("records"),
            columns=[{"name":c,"id":c} for c in dt.columns],
            page_size=25, sort_action="native", filter_action="native",
            style_table={"overflowX":"auto"},
            style_cell={"backgroundColor":CARD_BG,"color":TEXT,"border":f"1px solid {GRID}",
                        "fontSize":"13px","textAlign":"left","padding":"8px"},
            style_header={"backgroundColor":BG,"color":ACCENT,"fontWeight":"bold",
                          "border":f"1px solid {GRID}"},
            style_data_conditional=[{"if":{"row_index":0},"backgroundColor":"#2D1F1A","color":"#F97316"}],
        ))

    # ── Insights Exclusifs ────────────────────────────────────────────────────
    elif tab == "insights":
        solo_cats = ["HYROX MEN","HYROX WOMEN","HYROX PRO MEN","HYROX PRO WOMEN"]
        d_solo = d[d["Category"].astype(str).isin(solo_cats)].copy()

        # ── 1. Run Pace heatmap (top / médiane / bottom) ──────────────────────
        rc = [c for c in RUN_COLS if c in d.columns]
        fig_pace = go.Figure()
        levels = [
            ("Top 10%",    d_solo["Total_sec"].quantile(0.10), ACCENT,  "solid",  3),
            ("Médiane",    d_solo["Total_sec"].quantile(0.50), BLUE,    "solid",  2),
            ("Bottom 10%", d_solo["Total_sec"].quantile(0.90), SUBTEXT, "dot",    2),
        ]
        for lbl, thr, col, dash, width in levels:
            if lbl == "Top 10%":
                sub = d_solo[d_solo["Total_sec"] <= thr]
            elif lbl == "Médiane":
                lo = d_solo["Total_sec"].quantile(0.40)
                hi = d_solo["Total_sec"].quantile(0.60)
                sub = d_solo[(d_solo["Total_sec"] >= lo) & (d_solo["Total_sec"] <= hi)]
            else:
                sub = d_solo[d_solo["Total_sec"] >= thr]
            sub = sub.dropna(subset=rc)
            if len(sub) == 0: continue
            paces = [sub[c].median() for c in rc]
            fig_pace.add_trace(go.Scatter(
                x=RUN_LABELS[:len(paces)], y=[p/60 for p in paces],
                mode="lines+markers+text",
                name=lbl,
                text=[f"{int(p//60)}:{int(p%60):02d}" for p in paces],
                textposition="top center",
                line=dict(color=col, width=width, dash=dash),
                marker=dict(size=8)))
        fig_pace.add_annotation(
            x="Run 2", yref="paper", y=1.05,
            text="Run 2 = fastest (Ski Erg propulsion effect)",
            showarrow=False, font=dict(color=GREEN, size=11))
        fig_pace.update_layout(
            title="Run Pace 1→8 — Top 10% / Médiane / Bottom 10%",
            yaxis_title="Pace (min/km)", **CHART_LAYOUT)

        # ── 2. Roxzone Impact ─────────────────────────────────────────────────
        d_rox = d_solo.dropna(subset=["Roxzone_sec","Total_sec"]).copy()
        d_rox["Rox_min"] = d_rox["Roxzone_sec"] / 60
        d_rox["Total_min_r"] = d_rox["Total_sec"] / 60
        d_rox_s = d_rox.sample(min(3000, len(d_rox)), random_state=42)
        fig_rox = px.scatter(
            d_rox_s, x="Rox_min", y="Total_min_r",
            color="Category", opacity=0.5, size_max=6,
            labels={"Rox_min": "Roxzone (min)", "Total_min_r": "Temps final (min)"},
            title="Roxzone vs Temps final — chaque point = 1 athlète",
            color_discrete_sequence=COLORS)
        # Ligne de tendance manuelle
        if len(d_rox) > 10:
            z = np.polyfit(d_rox["Rox_min"].fillna(0), d_rox["Total_min_r"].fillna(0), 1)
            x_line = np.linspace(d_rox["Rox_min"].min(), d_rox["Rox_min"].max(), 100)
            fig_rox.add_trace(go.Scatter(
                x=x_line, y=np.polyval(z, x_line),
                mode="lines", name="Tendance",
                line=dict(color=ACCENT, width=2, dash="dash")))
        fig_rox.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.95,
            text=f"Top 10% = 5.4 min · Bottom 10% = 12.1 min",
            showarrow=False, font=dict(color=ORANGE, size=11),
            bgcolor=CARD_BG, bordercolor=ORANGE)
        fig_rox.update_layout(**CHART_LAYOUT)
        fig_rox.update_traces(marker=dict(size=4))

        # ── 3. Station DNA (coefficient de variation) ─────────────────────────
        cv_data = []
        for col, lbl in zip(WORKOUT_COLS, WORKOUT_LABELS):
            s = d_solo[col].dropna()
            if len(s) < 20: continue
            cv = s.std() / s.mean() * 100
            cv_data.append({"Station": lbl, "CV": round(cv, 1)})
        cv_df = pd.DataFrame(cv_data).sort_values("CV", ascending=True)
        colors_cv = [GREEN if v < 15 else ORANGE if v < 25 else ACCENT for v in cv_df["CV"]]

        fig_dna = go.Figure(go.Bar(
            y=cv_df["Station"],
            x=cv_df["CV"],
            orientation="h",
            marker=dict(color=colors_cv, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"{v:.1f}%" for v in cv_df["CV"]],
            textposition="outside",
            textfont=dict(color=TEXT, size=12),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>CV = %{x:.1f}%<extra></extra>",
            showlegend=False,
        ))

        fig_dna.add_vline(x=20, line_dash="dot",
                          line_color="rgba(156,163,175,0.4)", line_width=1.5)
        fig_dna.add_annotation(
            x=20, yref="paper", y=1.04,
            text="seuil clivant (20%)",
            showarrow=False,
            font=dict(color=SUBTEXT, size=10),
            xanchor="center")

        for i, (lbl, col) in enumerate([
            ("● Très clivant (> 25%)", ACCENT),
            ("● Clivant (> 20%)",      ORANGE),
            ("● Homogène",             GREEN),
        ]):
            fig_dna.add_annotation(
                xref="paper", yref="paper",
                x=1.0, y=0.05 + i * 0.08,
                text=lbl, showarrow=False,
                font=dict(color=col, size=10),
                xanchor="right", align="right")

        fig_dna.update_layout(
            title=dict(text="Quelle station crée le plus d'écart entre athlètes ?",
                       font=dict(size=14, color=TEXT)),
            xaxis=dict(
                title="Coefficient de variation (%)",
                range=[0, 38],
                gridcolor=GRID, zerolinecolor=GRID,
                tickvals=[0, 10, 20, 30],
                tickfont=dict(size=11),
            ),
            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=13)),
            margin=dict(l=130, r=20, t=55, b=50),
            height=370,
            showlegend=False,
            paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
            font=dict(color=TEXT, family="Inter, sans-serif"),
            hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT),
        )

        # ── 4. Battle of Formats ──────────────────────────────────────────────
        format_cats = [
            ("HYROX MEN",             "Solo Hommes"),
            ("HYROX DOUBLES MEN",     "Doubles Hommes"),
            ("HYROX PRO MEN",         "Pro Hommes"),
            ("HYROX WOMEN",           "Solo Femmes"),
            ("HYROX DOUBLES WOMEN",   "Doubles Femmes"),
            ("HYROX PRO WOMEN",       "Pro Femmes"),
        ]
        fig_fmt = go.Figure()
        for cat, lbl in format_cats:
            sub = d[d["Category"].astype(str) == cat]["Total_sec"].dropna() / 60
            if len(sub) < 10: continue
            col = ACCENT if "MEN" in cat else BLUE
            fig_fmt.add_trace(go.Box(
                y=sub, name=lbl,
                marker_color=col, boxmean=True,
                line=dict(color=col)))
        fig_fmt.update_layout(
            title="Battle of Formats — Distribution des temps par format (box = médiane + IQR)",
            yaxis_title="Temps final (min)", showlegend=False, **CHART_LAYOUT)

        # ── 5. Predictor Score ────────────────────────────────────────────────
        pred_data = []
        d_pred = d_solo.dropna(subset=["Total_sec"])
        for col, lbl in zip(WORKOUT_COLS, WORKOUT_LABELS):
            s = d_pred[[col, "Total_sec"]].dropna()
            if len(s) < 50: continue
            corr = s[col].corr(s["Total_sec"])
            pred_data.append({"Station": lbl, "Corrélation": round(corr, 3)})
        pred_df = pd.DataFrame(pred_data).sort_values("Corrélation", ascending=True)

        colors_pred = [ACCENT if v > 0.75 else ORANGE if v > 0.65 else GREEN
                       for v in pred_df["Corrélation"]]

        fig_pred = go.Figure(go.Bar(
            y=pred_df["Station"],
            x=pred_df["Corrélation"],
            orientation="h",
            marker=dict(color=colors_pred, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"r = {v:.3f}" for v in pred_df["Corrélation"]],
            textposition="outside",
            textfont=dict(color=TEXT, size=12),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>",
            showlegend=False,
        ))

        # Ligne seuil 0.75
        fig_pred.add_vline(x=0.75, line_dash="dot",
                           line_color="rgba(232,71,43,0.4)", line_width=1.5)

        # Légende couleurs via annotations dans le coin
        for i, (lbl, col) in enumerate([
            ("● Fort (> 0.75)",    ACCENT),
            ("● Modéré (> 0.65)",  ORANGE),
            ("● Faible",           GREEN),
        ]):
            fig_pred.add_annotation(
                xref="paper", yref="paper",
                x=1.0, y=0.05 + i * 0.08,
                text=lbl, showarrow=False,
                font=dict(color=col, size=10),
                xanchor="right", align="right")

        fig_pred.update_layout(
            title=dict(text="Quelle station prédit le mieux votre temps final ?",
                       font=dict(size=14, color=TEXT)),
            xaxis=dict(
                title="Corrélation de Pearson",
                range=[0, 1.15],
                gridcolor=GRID, zerolinecolor=GRID,
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                tickfont=dict(size=11),
            ),
            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=13)),
            margin=dict(l=130, r=20, t=55, b=50),
            height=370,
            showlegend=False,
            paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
            font=dict(color=TEXT, family="Inter, sans-serif"),
            hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT),
        )

        # ── Insight cards ─────────────────────────────────────────────────────
        insights_row = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("🎿", style={"fontSize":"2rem"}),
                html.Div("Ski Erg : score moyen 26.9/100", style={"fontWeight":"700","color":ACCENT}),
                html.Div("Station la moins maîtrisée par le field", style={"color":SUBTEXT,"fontSize":".8rem"}),
            ]), style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px","textAlign":"center"}), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("⏱️", style={"fontSize":"2rem"}),
                html.Div("Roxzone : 6min42 perdues en transitions", style={"fontWeight":"700","color":ORANGE}),
                html.Div("Top 10% = 5.4min · Bottom 10% = 12.1min", style={"color":SUBTEXT,"fontSize":".8rem"}),
            ]), style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px","textAlign":"center"}), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("🤸", style={"fontSize":"2rem"}),
                html.Div("Burpee BJ prédit le mieux le résultat", style={"fontWeight":"700","color":BLUE}),
                html.Div("Corrélation 0.813 avec le temps final", style={"color":SUBTEXT,"fontSize":".8rem"}),
            ]), style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px","textAlign":"center"}), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("👥", style={"fontSize":"2rem"}),
                html.Div("Doubles = 9.6min plus rapides que solo", style={"fontWeight":"700","color":GREEN}),
                html.Div("73.1min vs 82.7min (hommes)", style={"color":SUBTEXT,"fontSize":".8rem"}),
            ]), style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px","textAlign":"center"}), md=3),
        ], className="g-3 mb-4")

        return html.Div([
            insights_row,
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_pace, config={"displayModeBar":False})), md=7),
                dbc.Col(card(dcc.Graph(figure=fig_dna,  config={"displayModeBar":False})), md=5),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_rox,  config={"displayModeBar":False})), md=7),
                dbc.Col(card(dcc.Graph(figure=fig_pred, config={"displayModeBar":False})), md=5),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_fmt,  config={"displayModeBar":False})), md=12),
            ], className="g-3"),
        ])

    return html.Div("Chargement...", style={"color":SUBTEXT})

# ── CB4 : Profil athlète ──────────────────────────────────────────────────────
@app.callback(Output("ath-content","children"),
              [Input("ath-sel","value"), Input("ath-cmp","value")],
              State("store","data"), prevent_initial_call=True)
def upd_athlete(idx, cmp_mode, store):
    if idx is None: return html.Div("Sélectionnez un athlète.", style={"color":SUBTEXT})
    d = get_d(store); row = df.loc[idx]
    display_name = row.get("Team_Name") if pd.notna(row.get("Team_Name")) else row["Name"]
    short = str(display_name).split(",")[0].split()[-1]
    rank  = row.get("Rank","?"); rank_ag = row.get("Rank_AG","?")
    pct   = (1-(rank-1)/len(d))*100 if isinstance(rank,(int,float)) and len(d)>0 else 0

    if cmp_mode == "median":
        ref = d[WORKOUT_COLS].median(); ref_r = d[[c for c in RUN_COLS if c in d.columns]].median()
        ref_lbl = "Médiane"
    else:
        q10 = d["Total_sec"].quantile(0.10); top_d = d[d["Total_sec"]<=q10]
        ref = top_d[WORKOUT_COLS].mean() if len(top_d) else d[WORKOUT_COLS].median()
        ref_r = (top_d[[c for c in RUN_COLS if c in top_d.columns]].mean()
                 if len(top_d) else d[[c for c in RUN_COLS if c in d.columns]].median())
        ref_lbl = "Top 10%"

    ath_v = [row[c]/60 if pd.notna(row.get(c)) else 0 for c in WORKOUT_COLS]
    ref_v = [ref[c]/60 for c in WORKOUT_COLS]

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatterpolar(r=ath_v+[ath_v[0]], theta=WORKOUT_LABELS+[WORKOUT_LABELS[0]],
                                      fill="toself", name=short,
                                      line=dict(color=ACCENT), fillcolor="rgba(232,71,43,0.2)"))
    fig_rad.add_trace(go.Scatterpolar(r=ref_v+[ref_v[0]], theta=WORKOUT_LABELS+[WORKOUT_LABELS[0]],
                                      fill="toself", name=ref_lbl,
                                      line=dict(color=BLUE), fillcolor="rgba(59,130,246,0.15)"))
    fig_rad.update_layout(
        title=f"Radar — {display_name} vs {ref_lbl}",
        polar=dict(radialaxis=dict(visible=True,gridcolor=GRID,color=SUBTEXT),
                   bgcolor=PLOTBG, angularaxis=dict(color=SUBTEXT)),
        **{k:v for k,v in CHART_LAYOUT.items() if k not in ["xaxis","yaxis"]})

    rc = [c for c in RUN_COLS if c in df.columns]
    ath_r = [row[c]/60 if pd.notna(row.get(c)) else 0 for c in rc]
    ref_r2 = [ref_r[c]/60 for c in rc]
    fig_runs = go.Figure()
    fig_runs.add_trace(go.Bar(name=short, x=RUN_LABELS[:len(ath_r)], y=ath_r,
                              marker_color=ACCENT, text=[sec_to_mmss(v*60) for v in ath_r],
                              textposition="outside"))
    fig_runs.add_trace(go.Bar(name=ref_lbl, x=RUN_LABELS[:len(ref_r2)], y=ref_r2,
                              marker_color=BLUE, text=[sec_to_mmss(v*60) for v in ref_r2],
                              textposition="outside"))
    fig_runs.update_layout(barmode="group", title="Splits Running",
                           yaxis_title="Temps (min)", **CHART_LAYOUT)

    medians = {col: d[col].median() for col in WORKOUT_COLS if col in d.columns}
    score_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.Div(sec_to_mmss(row.get(WORKOUT_COLS[i])) if pd.notna(row.get(WORKOUT_COLS[i])) else "–",
                     style={"fontSize":"1.4rem","fontWeight":"700",
                            "color": GREEN if pd.notna(row.get(WORKOUT_COLS[i])) and
                                     row.get(WORKOUT_COLS[i]) <= medians.get(WORKOUT_COLS[i],99999)
                                     else ACCENT}),
            html.Div(lbl, style={"fontSize":".65rem","color":SUBTEXT}),
        ])], style={"backgroundColor":BG,"border":f"1px solid {GRID}","textAlign":"center"}), md=3)
        for i, lbl in enumerate(WORKOUT_LABELS)
    ], className="g-2")

    # Balance Score badge
    bal = row.get("Balance_Score")
    if not pd.isna(bal):
        run_pct = float(bal); work_pct = 100 - run_pct
        if run_pct > 55: lbl_b, col_b = "Profil Runner", "primary"
        elif run_pct < 45: lbl_b, col_b = "Profil Worker", "danger"
        else: lbl_b, col_b = "Profil Équilibré", "success"
        bal_row = dbc.Row([dbc.Col([
            html.Div("Score équilibre", style={"color":SUBTEXT,"fontSize":".75rem","textTransform":"uppercase","marginBottom":"6px"}),
            dbc.Progress([
                dbc.Progress(value=run_pct, color="primary", bar=True, label=f"Running {run_pct:.0f}%"),
                dbc.Progress(value=work_pct, color="danger", bar=True, label=f"Workout {work_pct:.0f}%"),
            ], style={"height":"22px","borderRadius":"6px"}),
            dbc.Badge(lbl_b, color=col_b, className="mt-2"),
        ])], className="mt-2")
    else:
        bal_row = html.Div()

    summary = dbc.Card([dbc.CardBody([
        html.H4(str(display_name), style={"color":TEXT,"fontWeight":"800"}),
        html.Div(f"{row.get('Category','')} · {row.get('Age_Group','')} · {row.get('Country','')}",
                 style={"color":SUBTEXT,"marginBottom":"12px"}),
        dbc.Row([
            dbc.Col([html.Div(row.get("Finish_Time","–"), style={"fontSize":"2rem","fontWeight":"800","color":ACCENT}),
                     html.Div("Temps final", style={"color":SUBTEXT,"fontSize":".75rem"})]),
            dbc.Col([html.Div(f"#{rank}", style={"fontSize":"1.8rem","fontWeight":"700","color":TEXT}),
                     html.Div("Classement", style={"color":SUBTEXT,"fontSize":".75rem"})]),
            dbc.Col([html.Div(f"#{rank_ag}", style={"fontSize":"1.8rem","fontWeight":"700","color":TEXT}),
                     html.Div("Classement AG", style={"color":SUBTEXT,"fontSize":".75rem"})]),
            dbc.Col([html.Div(f"Top {100-pct:.1f}%", style={"fontSize":"1.4rem","fontWeight":"700","color":GREEN}),
                     html.Div("Percentile", style={"color":SUBTEXT,"fontSize":".75rem"})]),
        ]),
        html.Hr(style={"borderColor":GRID}),
        html.Div("Temps par station", style={"color":SUBTEXT,"marginBottom":"8px","fontSize":".75rem","textTransform":"uppercase"}),
        score_cards,
        html.Hr(style={"borderColor":GRID}),
        bal_row,
    ])], style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px"})

    return html.Div([
        summary,
        html.Div(style={"height":"16px"}),
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig_rad,  config={"displayModeBar":False})), md=5),
            dbc.Col(card(dcc.Graph(figure=fig_runs, config={"displayModeBar":False})), md=7),
        ], className="g-3"),
    ])

# ── CB5 : Coach ───────────────────────────────────────────────────────────────
@app.callback(Output("coach-content","children"),
              Input("coach-sel","value"),
              State("store","data"), prevent_initial_call=True)
def upd_coach(idx, store):
    if idx is None: return html.Div("Sélectionnez un athlète.", style={"color":SUBTEXT})
    d = get_d(store); row = df.loc[idx]
    display_name = row.get("Team_Name") if pd.notna(row.get("Team_Name")) else row["Name"]
    first = str(display_name).split(",")[0].split()[0]
    rank  = row.get("Rank"); n = len(d)
    pct   = (1-(rank-1)/n)*100 if isinstance(rank,(int,float)) and n>0 else None
    cat   = row.get("Category",""); d_cat = df[df["Category"]==cat] if cat else d

    diag = dbc.Alert([html.I(className="bi bi-lightning-fill me-2"),
                      html.Strong(f"{first}, tu es dans le Top {100-pct:.0f}% "
                                  f"(#{rank} sur {n}) — voici tes marges.")],
                     color="warning", style={"backgroundColor":"#2D2000","borderColor":ORANGE,"color":TEXT}) \
           if pct is not None else html.Div()

    cat_med = {col: d_cat[col].median() for col in WORKOUT_COLS if col in d_cat.columns}
    gains = []
    for lbl, col in zip(WORKOUT_LABELS, WORKOUT_COLS):
        av = row.get(col); mv = cat_med.get(col)
        if pd.isna(av) or pd.isna(mv): continue
        gains.append({"Station":lbl,"Ton temps":av,"Médiane catégorie":mv,"Écart (s)":av-mv})
    gains_df = pd.DataFrame(gains).sort_values("Écart (s)", ascending=False)

    rows_t = []
    for _, g in gains_df.iterrows():
        ec = g["Écart (s)"]
        bg = "#3B0000" if ec>30 else "#2D1A00" if ec>0 else "#002B1A"
        tc = "#FCA5A5" if ec>30 else ORANGE if ec>0 else GREEN
        gs = f"-{sec_to_mmss(abs(int(ec)))}" if ec>0 else f"+{sec_to_mmss(abs(int(ec)))}"
        rows_t.append(html.Tr([
            html.Td(g["Station"], style={"color":TEXT,"padding":"8px 12px"}),
            html.Td(sec_to_mmss(g["Ton temps"]),         style={"color":ACCENT,"padding":"8px 12px","textAlign":"center"}),
            html.Td(sec_to_mmss(g["Médiane catégorie"]), style={"color":SUBTEXT,"padding":"8px 12px","textAlign":"center"}),
            html.Td(gs, style={"color":tc,"padding":"8px 12px","textAlign":"center",
                               "fontWeight":"700","backgroundColor":bg,"borderRadius":"4px"}),
        ], style={"borderBottom":f"1px solid {GRID}"}))

    table = dbc.Card([dbc.CardBody([
        html.H6("📊 Gains potentiels", style={"color":ACCENT,"textTransform":"uppercase","letterSpacing":".08em"}),
        html.Table([
            html.Thead(html.Tr([html.Th(h, style={"color":SUBTEXT,"padding":"8px 12px"})
                                for h in ["Station","Ton temps","Médiane catégorie","Écart"]]),
                       style={"borderBottom":f"2px solid {ACCENT}"}),
            html.Tbody(rows_t),
        ], style={"width":"100%","borderCollapse":"collapse"}),
    ])], style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px"})

    insights = []
    if len(gains_df) > 0:
        worst = gains_df.iloc[0]
        if worst["Écart (s)"] > 0:
            delta_s = int(worst["Écart (s)"])
            sim = row.get("Total_sec",0) - delta_s
            gp  = int((d["Total_sec"]<row.get("Total_sec",0)).sum() - (d["Total_sec"]<sim).sum())
            insights.append(dbc.Alert([
                html.I(className="bi bi-bullseye me-2"),
                html.Strong("Insight : "),
                f"{worst['Station']} — {sec_to_mmss(int(worst['Ton temps']))} vs médiane {sec_to_mmss(int(worst['Médiane catégorie']))} "
                f"→ {sec_to_mmss(delta_s)} de perdu. En progressant là, tu gagnes ~{abs(gp)} place(s)."
            ], color="danger", style={"backgroundColor":"#2D0A00","borderColor":ACCENT,"color":TEXT}))

    run_t = row.get("Runs_Total_sec"); work_t = row.get("Workouts_Total_sec")
    if pd.notna(run_t) and pd.notna(work_t) and len(d_cat)>1:
        rp = (d_cat["Runs_Total_sec"]<run_t).mean()*100
        wp = (d_cat["Workouts_Total_sec"]<work_t).mean()*100
        diff = rp - wp
        if diff>5: pt,pc = f"Runner (top {100-rp:.0f}% run vs top {100-wp:.0f}% workout)", BLUE
        elif diff<-5: pt,pc = f"Worker (top {100-rp:.0f}% run vs top {100-wp:.0f}% workout)", GREEN
        else: pt,pc = f"Équilibré (top {100-rp:.0f}% run vs top {100-wp:.0f}% workout)", ORANGE
        insights.append(dbc.Alert([html.I(className="bi bi-person-fill-check me-2"),
                                   html.Strong("Profil : "), pt],
                                  style={"backgroundColor":CARD_BG,"borderColor":pc,"color":TEXT,
                                         "borderLeft":f"4px solid {pc}"}))

    return html.Div([diag, table, html.Div(style={"height":"12px"}), html.Div(insights)])

# ── CB6 : Face-à-Face ─────────────────────────────────────────────────────────
@app.callback(Output("ff-content","children"),
              [Input("ff-a","value"), Input("ff-b","value")],
              State("store","data"), prevent_initial_call=True)
def upd_ff(idx_a, idx_b, store):
    if idx_a is None or idx_b is None:
        return html.Div("Sélectionnez deux athlètes.", style={"color":SUBTEXT,"padding":"20px"})
    if idx_a == idx_b:
        return dbc.Alert("Choisissez deux athlètes différents.", color="warning",
                         style={"backgroundColor":"#2D2000","color":TEXT,"borderColor":ORANGE})
    d = get_d(store); ra = df.loc[idx_a]; rb = df.loc[idx_b]
    na = ra.get("Team_Name") if pd.notna(ra.get("Team_Name")) else ra["Name"]
    nb = rb.get("Team_Name") if pd.notna(rb.get("Team_Name")) else rb["Name"]
    sa = str(na).split(",")[0].split()[-1]; sb = str(nb).split(",")[0].split()[-1]
    rka = ra.get("Rank","?"); rkb = rb.get("Rank","?"); n = len(d)
    pa  = (1-(rka-1)/n)*100 if isinstance(rka,(int,float)) and n>0 else None
    pb  = (1-(rkb-1)/n)*100 if isinstance(rkb,(int,float)) and n>0 else None

    def stat_row(lbl, va, vb, fmt=lambda x:x, lib=True):
        fa,fb = fmt(va),fmt(vb)
        ca=cb=TEXT
        if pd.notna(va) and pd.notna(vb):
            ca = GREEN if (va<vb)==lib else (ACCENT if va!=vb else TEXT)
            cb = GREEN if (vb<va)==lib else (ACCENT if va!=vb else TEXT)
        return dbc.Row([
            dbc.Col(html.Div(fa,style={"color":ca,"fontWeight":"700","fontSize":"1.1rem","textAlign":"right"}),md=4),
            dbc.Col(html.Div(lbl,style={"color":SUBTEXT,"fontSize":".7rem","textAlign":"center","textTransform":"uppercase"}),md=4),
            dbc.Col(html.Div(fb,style={"color":cb,"fontWeight":"700","fontSize":"1.1rem","textAlign":"left"}),md=4),
        ], className="mb-1")

    summary = dbc.Card([dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Div([html.Div(str(na).upper(),style={"color":ACCENT,"fontWeight":"800","fontSize":"1.1rem"}),
                              html.Div(f"{ra.get('Category','')} · {ra.get('Age_Group','')}",style={"color":SUBTEXT,"fontSize":".75rem"})],
                             style={"textAlign":"right"}),md=5),
            dbc.Col(html.Div("⚔️",style={"textAlign":"center","fontSize":"1.5rem"}),md=2),
            dbc.Col(html.Div([html.Div(str(nb).upper(),style={"color":BLUE,"fontWeight":"800","fontSize":"1.1rem"}),
                              html.Div(f"{rb.get('Category','')} · {rb.get('Age_Group','')}",style={"color":SUBTEXT,"fontSize":".75rem"})]),md=5),
        ], className="mb-3 align-items-center"),
        html.Hr(style={"borderColor":GRID}),
        stat_row("Temps final", ra.get("Total_sec"),         rb.get("Total_sec"),         sec_to_mmss),
        stat_row("Classement",  rka,                          rkb,                          lambda x:f"#{x}"),
        stat_row("Percentile",  pa,                           pb,                           lambda x:f"Top {100-x:.0f}%" if x else "–", lib=False),
        stat_row("Running",     ra.get("Runs_Total_sec"),     rb.get("Runs_Total_sec"),     sec_to_mmss),
        stat_row("Workout",     ra.get("Workouts_Total_sec"), rb.get("Workouts_Total_sec"), sec_to_mmss),
        stat_row("Roxzone",     ra.get("Roxzone_sec"),        rb.get("Roxzone_sec"),        sec_to_mmss),
    ])], style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px"})

    # Radar stations
    va = [ra.get(c,0) or 0 for c in WORKOUT_COLS]
    vb = [rb.get(c,0) or 0 for c in WORKOUT_COLS]
    mn,mx = min(va+vb),max(va+vb)
    na2=[((v-mn)/(mx-mn+1))*10 for v in va]; nb2=[((v-mn)/(mx-mn+1))*10 for v in vb]
    fig_r=go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=na2+[na2[0]],theta=WORKOUT_LABELS+[WORKOUT_LABELS[0]],
                                    fill="toself",name=sa,line=dict(color=ACCENT),fillcolor="rgba(232,71,43,0.2)"))
    fig_r.add_trace(go.Scatterpolar(r=nb2+[nb2[0]],theta=WORKOUT_LABELS+[WORKOUT_LABELS[0]],
                                    fill="toself",name=sb,line=dict(color=BLUE),fillcolor="rgba(59,130,246,0.15)"))
    fig_r.update_layout(title=f"Stations — {sa} vs {sb}",
                        polar=dict(radialaxis=dict(visible=False,gridcolor=GRID,color=SUBTEXT),
                                   bgcolor=PLOTBG,angularaxis=dict(color=SUBTEXT)),
                        **{k:v for k,v in CHART_LAYOUT.items() if k not in ["xaxis","yaxis"]})

    fig_st=go.Figure()
    fig_st.add_trace(go.Bar(name=sa,x=WORKOUT_LABELS,y=[v/60 for v in va],marker_color=ACCENT,
                            text=[sec_to_mmss(v) for v in va],textposition="outside"))
    fig_st.add_trace(go.Bar(name=sb,x=WORKOUT_LABELS,y=[v/60 for v in vb],marker_color=BLUE,
                            text=[sec_to_mmss(v) for v in vb],textposition="outside"))
    fig_st.update_layout(barmode="group",title="Stations",yaxis_title="min",**CHART_LAYOUT)

    rc=[c for c in RUN_COLS if c in df.columns]
    ra_r=[ra.get(c,0) or 0 for c in rc]; rb_r=[rb.get(c,0) or 0 for c in rc]
    fig_ru=go.Figure()
    fig_ru.add_trace(go.Bar(name=sa,x=RUN_LABELS[:len(ra_r)],y=[v/60 for v in ra_r],marker_color=ACCENT,
                            text=[sec_to_mmss(v) for v in ra_r],textposition="outside"))
    fig_ru.add_trace(go.Bar(name=sb,x=RUN_LABELS[:len(rb_r)],y=[v/60 for v in rb_r],marker_color=BLUE,
                            text=[sec_to_mmss(v) for v in rb_r],textposition="outside"))
    fig_ru.update_layout(barmode="group",title="Runs",yaxis_title="min",**CHART_LAYOUT)

    wins_a=wins_b=0
    vrd_rows=[]
    for lbl,col in zip(WORKOUT_LABELS,WORKOUT_COLS):
        va2=ra.get(col); vb2=rb.get(col)
        if pd.notna(va2) and pd.notna(vb2):
            w=sa if va2<vb2 else sb; wc=ACCENT if va2<vb2 else BLUE
            if va2<vb2: wins_a+=1
            else: wins_b+=1
            vrd_rows.append(html.Tr([
                html.Td(lbl,style={"color":TEXT,"padding":"6px 12px"}),
                html.Td(sec_to_mmss(va2),style={"color":ACCENT,"padding":"6px 12px","textAlign":"center"}),
                html.Td(sec_to_mmss(vb2),style={"color":BLUE,"padding":"6px 12px","textAlign":"center"}),
                html.Td(f"✓ {w}",style={"color":wc,"padding":"6px 12px","textAlign":"center","fontWeight":"700"}),
            ],style={"borderBottom":f"1px solid {GRID}"}))

    wv=sa if wins_a>wins_b else (sb if wins_b>wins_a else "Égalité")
    wvc=ACCENT if wins_a>wins_b else (BLUE if wins_b>wins_a else ORANGE)
    verdict=dbc.Card([dbc.CardBody([
        html.H6("🏆 Verdict station par station",style={"color":ACCENT,"textTransform":"uppercase","letterSpacing":".08em"}),
        html.Table([
            html.Thead(html.Tr([html.Th(h,style={"color":SUBTEXT,"padding":"6px 12px"})
                                for h in ["Station",sa,sb,"Vainqueur"]]),style={"borderBottom":f"2px solid {ACCENT}"}),
            html.Tbody(vrd_rows),
        ],style={"width":"100%","borderCollapse":"collapse"}),
        html.Hr(style={"borderColor":GRID}),
        dbc.Alert([html.Strong(f"🏆 {wv} "),html.Span(f"({wins_a} vs {wins_b} disciplines)")],
                  style={"backgroundColor":CARD_BG,"borderColor":wvc,"color":wvc,
                         "fontWeight":"700","fontSize":"1.1rem","borderLeft":f"4px solid {wvc}"}),
    ])],style={"backgroundColor":CARD_BG,"border":f"1px solid {GRID}","borderRadius":"10px"})

    return html.Div([
        summary, html.Div(style={"height":"16px"}),
        dbc.Row([dbc.Col(card(dcc.Graph(figure=fig_r, config={"displayModeBar":False})),md=5),
                 dbc.Col(card(dcc.Graph(figure=fig_st,config={"displayModeBar":False})),md=7)],
                className="g-3 mb-3"),
        dbc.Row([dbc.Col(card(dcc.Graph(figure=fig_ru,config={"displayModeBar":False})),md=12)],
                className="g-3 mb-3"),
        verdict,
    ])

# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n✅  Dashboard Hyrox Paris 2026")
    print("🌐  Ouvre http://127.0.0.1:8051 dans ton navigateur\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8051)))
