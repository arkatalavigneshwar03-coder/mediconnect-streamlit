import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import SEGS, pretty
from src.models import regression_metrics
from src.ui import get_models, hbar, page_setup, seg_color, sidebar_note, style

page_setup("Regression", "💰")
sidebar_note()
M, _ = get_models()
df = M.survey
y, p_r, p_g = M.reg_test["y"], M.reg_test["p_ridge"], M.reg_test["p_gbr"]

st.title("💰 Regression: how much will they spend each month?")
st.markdown(f"Target: **Q31** maximum monthly spend in AED. Inputs: Q1–Q28, the Section E services, Q29 intent and "
            f"Q32 plan. Tested on {len(y)} held-out responses.")

m_r, m_g = regression_metrics(y, p_r), regression_metrics(y, p_g)
c = st.columns(4)
c[0].metric("R² (gradient boosting)", f"{m_g['R²']:.3f}", f"ridge {m_r['R²']:.3f}", delta_color="off")
c[1].metric("Mean error", f"{m_g['MAE (AED)']:.1f} AED", f"ridge {m_r['MAE (AED)']:.1f}", delta_color="off")
c[2].metric("Average spend (all)", f"{df.wtp_monthly_aed.mean():.1f} AED")
c[3].metric("Would pay nothing", f"{(df.wtp_monthly_aed == 0).mean():.0%}")

left, mid, right = st.columns(3)
with left:
    st.markdown("**Actual vs predicted** (gradient boosting)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=p_g, mode="markers", name="Held-out respondents",
                             marker=dict(color="#2a78d6", opacity=0.5, size=6),
                             hovertemplate="Actual %{x} AED<br>Predicted %{y:.0f} AED<extra></extra>"))
    mx = float(max(y.max(), p_g.max()))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect prediction",
                             line=dict(color="#7a8491", dash="dash")))
    fig.update_xaxes(title="Actual AED / month")
    fig.update_yaxes(title="Predicted AED / month")
    st.plotly_chart(style(fig, 380), use_container_width=True)
with mid:
    st.markdown("**Spend drivers** (AED/month per standard deviation, ridge)")
    coef = pd.DataFrame({"f": [pretty(n) for n in M.reg_names], "c": M.ridge.coef_})
    coef = coef.reindex(coef.c.abs().sort_values(ascending=False).index).head(12)
    st.plotly_chart(hbar(coef.f.tolist(), coef.c.round(2).tolist(), text=[f"{v:+.1f}" for v in coef.c],
                         colors=["#2a78d6" if v > 0 else "#d03b3b" for v in coef.c]), use_container_width=True)
with right:
    st.markdown("**Residuals** (actual − predicted)")
    fig = go.Figure(go.Histogram(x=y - p_g, marker_color="#2a78d6", xbins=dict(size=5)))
    fig.update_xaxes(title="AED")
    fig.update_yaxes(title="Respondents")
    st.plotly_chart(style(fig, 380, legend=False), use_container_width=True)

st.subheader("Pricing: how many likely subscribers would pay each price?")
st.caption("From Q31 answers of respondents who said 4–5 on Q29. Revenue index = price × share willing to pay.")
intd = df[df.interested == 1].wtp_monthly_aed.values
prices = np.arange(10, 121, 1)
share = np.array([(intd >= p).mean() for p in prices])
rev = prices * share
best = int(prices[rev.argmax()])
pc = st.columns([1, 1])
with pc[0]:
    fig = go.Figure(go.Scatter(x=prices, y=share, mode="lines", line=dict(color="#2a78d6", width=2),
                               hovertemplate="AED %{x}: %{y:.0%} would pay<extra></extra>"))
    fig.update_xaxes(title="Monthly price, AED")
    fig.update_yaxes(title="Share of likely subscribers", tickformat=".0%")
    st.plotly_chart(style(fig, 300, legend=False), use_container_width=True)
with pc[1]:
    fig = go.Figure(go.Scatter(x=prices, y=rev, mode="lines", line=dict(color="#0ca30c", width=2),
                               hovertemplate="AED %{x}: index %{y:.1f}<extra></extra>"))
    fig.add_vline(x=best, line_dash="dash", line_color="#7a8491", annotation_text=f"Peak at AED {best}")
    fig.update_xaxes(title="Monthly price, AED")
    fig.update_yaxes(title="Revenue per likely subscriber, AED")
    st.plotly_chart(style(fig, 300, legend=False), use_container_width=True)
st.caption("This ignores costs and churn; see Live Operations for the full profit picture.")

st.subheader("Average spend by group")
g1, g2, g3 = st.columns(3)
bs = df.groupby("segment").wtp_monthly_aed.mean().reindex(SEGS)
g1.plotly_chart(hbar(bs.index.tolist(), bs.round(1).tolist(), text=[f"{v:.0f} AED" for v in bs],
                     colors=[seg_color(s) for s in bs.index], xtitle="By segment"), use_container_width=True)
bi = df.groupby("income_band").wtp_monthly_aed.mean()
inc = {1: "Under 5k", 2: "5k–10k", 3: "10k–20k", 4: "20k–35k", 5: "35k+"}
g2.plotly_chart(hbar([inc[i] for i in bi.index], bi.round(1).tolist(), text=[f"{v:.0f} AED" for v in bi],
                     xtitle="By household income (AED/month)"), use_container_width=True)
bp = df.groupby("preferred_plan").wtp_monthly_aed.mean().sort_values(ascending=False)
g3.plotly_chart(hbar(bp.index.tolist(), bp.round(1).tolist(), text=[f"{v:.0f} AED" for v in bp],
                     xtitle="By preferred plan (Q32)"), use_container_width=True)

st.subheader("How likely subscribers want to buy")
b1, b2, b3 = st.columns(3)
for col, field, title in [(b1, "preferred_plan", "Plan (Q32)"), (b2, "purchase_channel", "Sign-up channel (Q33)"),
                          (b3, "payer", "Who pays (Q34)")]:
    ct = pd.crosstab(df[df.interested == 1][field], df[df.interested == 1].segment, normalize="columns").reindex(columns=SEGS)
    ct = ct.loc[ct.mean(axis=1).sort_values(ascending=True).index]
    fig = go.Figure([go.Bar(y=ct.index, x=ct[s], name=s, orientation="h", marker_color=seg_color(s),
                            hovertemplate="%{y}: %{x:.0%}<extra>" + s + "</extra>") for s in SEGS])
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%", title=title)
    col.plotly_chart(style(fig, 380), use_container_width=True)
