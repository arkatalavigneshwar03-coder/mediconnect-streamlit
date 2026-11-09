"""Streamlit helpers shared by every page: cached models, chart styling, small widgets."""
import plotly.graph_objects as go
import streamlit as st

from .config import GOOD, BAD, SEG_COLORS, SEGS
from .models import load_data, train_all
from .ops_sim import build_pool


@st.cache_resource(show_spinner="Training the four models on 1,500 survey responses…")
def get_models():
    survey, live = load_data()
    M = train_all(survey)
    return M, live


@st.cache_resource
def get_pool():
    M, _ = get_models()
    return build_pool(M.survey)


def page_setup(title: str, icon: str = "💊"):
    st.set_page_config(page_title=f"{title} · MediConnect", page_icon=icon, layout="wide")
    st.markdown("""
        <style>
          .block-container {padding-top: 2rem; padding-bottom: 3rem;}
          div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
          .small-note {color: #7a8491; font-size: 0.85rem;}
        </style>""", unsafe_allow_html=True)


def style(fig: go.Figure, height=340, legend=True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    return fig


def seg_color(i_or_name):
    name = SEGS[i_or_name] if isinstance(i_or_name, int) else i_or_name
    return SEG_COLORS.get(name, "#2a78d6")


def hbar(labels, values, text=None, colors=None, height=None, xtitle=None) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", text=text, textposition="outside",
                           marker_color=colors or "#2a78d6", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,.3~f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=xtitle, showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return style(fig, height or max(220, 34 * len(labels) + 60), legend=False)


def money(v, decimals=0):
    sign = "−" if v < 0 else ""
    return f"{sign}{abs(v):,.{decimals}f}"


def sidebar_note():
    st.sidebar.markdown(
        "<div class='small-note'>Synthetic data generated for the MediConnect survey design. "
        "Figures illustrate the method, not the real market.</div>", unsafe_allow_html=True)


__all__ = ["get_models", "get_pool", "page_setup", "style", "seg_color", "hbar", "money", "sidebar_note", "GOOD", "BAD"]
