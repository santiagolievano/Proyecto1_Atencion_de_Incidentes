
import pandas as pd
import numpy as np
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Carga 
df = pd.read_csv("https://raw.githubusercontent.com/santiagolievano/Proyecto1_Atencion_de_Incidentes/main/incident_event_log.csv")

# Fechas
for col in ["opened_at","resolved_at","closed_at"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

df["tiempo_resolucion_horas"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600

# Limpieza básica
df = df[~df["tiempo_resolucion_horas"].isna() & (df["tiempo_resolucion_horas"] >= 0)].copy()

# Mapeos ordinales
def norm(s):
    return str(s).strip().lower() if pd.notna(s) else s

if "priority" in df.columns:
    df["priority_norm"] = df["priority"].apply(norm)
else:
    df["priority_norm"] = np.nan

if "urgency" in df.columns:
    df["urgency_norm"]  = df["urgency"].apply(norm)
else:
    df["urgency_norm"] = np.nan

if "impact" in df.columns:
    df["impact_norm"]   = df["impact"].apply(norm)
else:
    df["impact_norm"] = np.nan

pri_map = {"1 - critical":4, "2 - high":3, "3 - moderate":2, "3 - medium":2, "4 - low":1}
urg_map = {"1 - high":3, "2 - medium":2, "3 - low":1}
imp_map = {"1 - high":3, "2 - medium":2, "3 - low":1}

df["priority_num"] = df["priority_norm"].map(pri_map) if "priority" in df.columns else np.nan
df["urgency_num"]  = df["urgency_norm"].map(urg_map)  if "urgency" in df.columns else np.nan
df["impact_num"]   = df["impact_norm"].map(imp_map)   if "impact" in df.columns else np.nan

# Top grupos
top_groups = df["assignment_group"].value_counts().head(15).index.tolist() if "assignment_group" in df.columns else []

#figuras

def fig_histograma(df_f):
    if df_f.empty:
        return go.Figure()
    lim = np.nanpercentile(df_f["tiempo_resolucion_horas"], 99)
    fig = px.histogram(df_f, x="tiempo_resolucion_horas", nbins=60)
    fig.update_xaxes(range=[0, lim], title="Horas")
    fig.update_yaxes(title="Frecuencia")
    fig.update_layout(title="Distribución del tiempo de resolución (horas)")
    return fig

def fig_box(df_f, by="priority"):
    if by not in df_f.columns or df_f.empty:
        return go.Figure()
    order = None
    if by == "priority":
        order = ["1 - Critical","2 - High","3 - Moderate","4 - Low"]
    if by == "urgency":
        order = ["1 - High","2 - Medium","3 - Low"]
    fig = px.box(df_f, x=by, y="tiempo_resolucion_horas", points=False, category_orders={by: order} if order else {})
    fig.update_yaxes(title="Horas", range=[0, np.nanpercentile(df_f['tiempo_resolucion_horas'],95)])
    fig.update_layout(title=f"Tiempo de resolución por {by}")
    return fig

def fig_sla_bar(df_f, by="assignment_group"):
    if "made_sla" not in df_f.columns or by not in df_f.columns or df_f.empty:
        return go.Figure()
    tmp = df_f.groupby(by)["made_sla"].mean().sort_values(ascending=False).reset_index().head(15)
    fig = px.bar(tmp, x="made_sla", y=by, orientation="h", labels={"made_sla":"% Cumplimiento SLA"})
    fig.update_layout(title=f"Porcentaje de cumplimiento SLA por {by} (Top 15)")
    return fig

def kpis(df_f):
    if df_f.empty:
        return np.nan, np.nan, np.nan
    mean_h = df_f["tiempo_resolucion_horas"].mean()
    median_h = df_f["tiempo_resolucion_horas"].median()
    sla = df_f["made_sla"].mean()*100 if "made_sla" in df_f.columns else np.nan
    return mean_h, median_h, sla

# Modelo predictivo rápido (regresión)

def preparar_features(df_in):
    feats = pd.DataFrame(index=df_in.index)
    for col in ["priority_num","urgency_num","impact_num","reassignment_count","reopen_count"]:
        if col in df_in.columns:
            feats[col] = df_in[col]
    # Dummies de grupos top
    if "assignment_group" in df_in.columns and top_groups:
        for g in top_groups[:10]:
            feats[f"grp_{g}"] = (df_in["assignment_group"]==g).astype(int)
    y = df_in["tiempo_resolucion_horas"]
    mask = feats.notna().all(axis=1) & y.notna()
    return feats.loc[mask], y.loc[mask]

X_all, y_all = preparar_features(df)

# --- FIX de muestreo: siempre devolver (X_sample, y_sample) como tupla ---
if len(X_all) > 10000:
    idx = X_all.sample(n=10000, random_state=42).index
    Xs = X_all.loc[idx]
    ys = y_all.loc[idx]
else:
    Xs, ys = X_all, y_all

if len(Xs) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.25, random_state=42)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred = lin.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    coef_series = pd.Series(lin.coef_, index=X_train.columns).sort_values(key=lambda s: s.abs(), ascending=False)
else:
    # Escenarios con muy pocos datos
    X_train = X_test = Xs
    y_train = y_test = ys
    y_pred = ys if len(ys)>0 else np.array([])
    r2 = np.nan
    mae = np.nan
    coef_series = pd.Series(dtype=float)

def fig_pred_vs_real(y_true, y_hat, r2_value, mae_value):
    if len(y_true)==0:
        return go.Figure()
    dfp = pd.DataFrame({"Real": y_true, "Predicho": y_hat})
    sample_n = min(len(dfp), 5000)
    dfp = dfp.sample(sample_n, random_state=42) if sample_n>0 else dfp
    fig = px.scatter(dfp, x="Real", y="Predicho", opacity=0.4)
    title_bits = []
    if not np.isnan(r2_value):
        title_bits.append(f"R²={r2_value:.3f}")
    if not np.isnan(mae_value):
        title_bits.append(f"MAE={mae_value:.1f} h")
    title_str = " | ".join(title_bits) if title_bits else "Sin métricas"
    fig.update_layout(title=f"Real vs Predicho ({title_str})")
    return fig

def fig_importancias(coefs):
    if coefs.empty:
        return go.Figure()
    tmp = coefs.head(15).sort_values()
    fig = px.bar(x=tmp.values, y=tmp.index, orientation="h", labels={"x":"Coeficiente (magnitud)","y":"Variable"})
    fig.update_layout(title="Importancia/Coeficientes (top 15) – Regresión lineal")
    return fig

#Dash

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Tablero – Gestión de Incidentes"),
    html.Div([
        html.Div([
            html.Label("Filtrar por prioridad:"),
            dcc.Dropdown(
                id="f-priority",
                options=[{"label":v, "value":v} for v in sorted(df["priority"].dropna().unique())] if "priority" in df.columns else [],
                multi=True
            )
        ], style={"width":"24%","display":"inline-block","verticalAlign":"top"}),
        html.Div([
            html.Label("Filtrar por urgencia:"),
            dcc.Dropdown(
                id="f-urgency",
                options=[{"label":v, "value":v} for v in sorted(df["urgency"].dropna().unique())] if "urgency" in df.columns else [],
                multi=True
            )
        ], style={"width":"24%","display":"inline-block","verticalAlign":"top","marginLeft":"1%"}),
        html.Div([
            html.Label("Filtrar por grupo (Top 15):"),
            dcc.Dropdown(
                id="f-group",
                options=[{"label":g, "value":g} for g in top_groups],
                multi=True
            )
        ], style={"width":"50%","display":"inline-block","verticalAlign":"top","marginLeft":"1%"}),
    ], style={"marginBottom":"15px"}),

    html.Div(id="kpis", style={"display":"flex","gap":"20px","marginBottom":"10px"}),

    dcc.Tabs([
        dcc.Tab(label="Descriptivo tiempos", children=[
            html.Div([
                dcc.Graph(id="hist-tiempo"),
                html.Div([
                    html.Div([dcc.Graph(id="box-priority")], style={"width":"49%","display":"inline-block"}),
                    html.Div([dcc.Graph(id="box-urgency")], style={"width":"49%","display":"inline-block"}),
                ])
            ])
        ]),
        dcc.Tab(label="SLA", children=[
            html.Div([
                html.Div([
                    html.Label("Desagregar SLA por:"),
                    dcc.Dropdown(
                        id="sla-by",
                        options=[{"label":"Grupo asignado","value":"assignment_group"},
                                 {"label":"Prioridad","value":"priority"},
                                 {"label":"Urgencia","value":"urgency"}],
                        value="assignment_group"
                    ),
                ], style={"width":"30%"}),
                dcc.Graph(id="bar-sla")
            ])
        ]),
        dcc.Tab(label="Predicción", children=[
            html.Div([
                dcc.Graph(id="pred-vs-real"),
                dcc.Graph(id="importancias")
            ])
        ]),
    ])
])

def aplicar_filtros(base, priority, urgency, group):
    df_f = base.copy()
    if priority and "priority" in df_f.columns:
        df_f = df_f[df_f["priority"].isin(priority)]
    if urgency and "urgency" in df_f.columns:
        df_f = df_f[df_f["urgency"].isin(urgency)]
    if group and "assignment_group" in df_f.columns:
        df_f = df_f[df_f["assignment_group"].isin(group)]
    return df_f

@callback(
    Output("kpis", "children"),
    Output("hist-tiempo", "figure"),
    Output("box-priority", "figure"),
    Output("box-urgency", "figure"),
    Output("bar-sla", "figure"),
    Output("pred-vs-real", "figure"),
    Output("importancias", "figure"),
    Input("f-priority","value"),
    Input("f-urgency","value"),
    Input("f-group","value"),
    Input("sla-by","value"),
)
def actualizar(priority, urgency, group, sla_by):
    df_f = aplicar_filtros(df, priority, urgency, group)

    mean_h, median_h, sla = kpis(df_f)
    kpi_cards = [
        html.Div([html.H4("Media (h)"), html.H3(f"{mean_h:.1f}" if not np.isnan(mean_h) else "N/A")], style={"border":"1px solid #ddd","padding":"10px","borderRadius":"8px","width":"200px"}),
        html.Div([html.H4("Mediana (h)"), html.H3(f"{median_h:.1f}" if not np.isnan(median_h) else "N/A")], style={"border":"1px solid #ddd","padding":"10px","borderRadius":"8px","width":"200px"}),
        html.Div([html.H4("% SLA cumplido"), html.H3(f"{sla:.1f}%" if not np.isnan(sla) else "N/A")], style={"border":"1px solid #ddd","padding":"10px","borderRadius":"8px","width":"200px"}),
    ]



    fig_hist = fig_histograma(df_f)
    fig_box_p = fig_box(df_f, by="priority") if "priority" in df_f.columns else go.Figure()
    fig_box_u = fig_box(df_f, by="urgency") if "urgency" in df_f.columns else go.Figure()
    by_col = sla_by if sla_by in df_f.columns else ("assignment_group" if "assignment_group" in df_f.columns else None)
    fig_sla = fig_sla_bar(df_f, by=by_col) if by_col else go.Figure()

    fig_pvr = fig_pred_vs_real(y_test, y_pred, r2, mae)
    fig_imp = fig_importancias(coef_series)

    return kpi_cards, fig_hist, fig_box_p, fig_box_u, fig_sla, fig_pvr, fig_imp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
