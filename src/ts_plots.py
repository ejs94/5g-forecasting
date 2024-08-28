from itertools import cycle

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"


def plot_forecast(actual_ts, forecast_ts, title):
    actual_df = actual_ts.pd_dataframe()
    forecast_df = forecast_ts.pd_dataframe()

    colors = [
        c.replace("rgb", "rgba").replace(")", ", <alpha>)")
        for c in px.colors.qualitative.Dark2
    ]

    act_color = colors[0]
    colors = cycle(colors[1:])
    dash_types = cycle(["dash", "dot", "dashdot"])

    # Criar subplots
    fig = make_subplots(
        rows=len(actual_df.columns),
        cols=1,
        shared_xaxes=True,
        subplot_titles=actual_df.columns,
    )

    # Adicionar valores reais e previstos em subplots separados
    for i, column in enumerate(actual_df.columns, start=1):
        fig.add_trace(
            go.Scatter(
                x=actual_df.index,
                y=actual_df[column],
                mode="lines",
                line=dict(color=act_color.replace("<alpha>", "0.3")),
                name=f"Actual {column}",
            ),
            row=i,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df[column],
                mode="lines",
                line=dict(
                    dash=next(dash_types), color=next(colors).replace("<alpha>", "1")
                ),
                name=f"Forecast {column}",
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        title=title,
        showlegend=True,  # Para evitar legendas duplicadas
        height=300
        * len(
            actual_df.columns
        ),  # Ajustar altura do gráfico conforme o número de subplots
    )

    # Atualizar eixos x para incluir toda a série e a previsão e formatar apenas a hora
    fig.update_xaxes(tickformat="%H:%M:%S")  # Formatação para mostrar apenas a hora

    return fig
