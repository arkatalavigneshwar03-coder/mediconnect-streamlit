"""Live simulation: 300 new respondents arrive and are scored by all four models as they come in."""
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AGE_LABELS, SEGS
from src.models import classification_metrics, next_offer, pca_coords, predict_segment, predict_subscribe, predict_wtp
from src.ui import get_models, hbar, page_setup, seg_color, sidebar_note, style

page_setup("Live Survey Stream", "📡")
sidebar_note()
M, live = get_models()
N = len(live)
ss = st.session_state
TEST_ACC = classification_metrics(M.cls_test["y"], M.cls_test["p_logit"])["Accuracy"]
SURVEY_MIX = M.survey.segment.value_counts(normalize=True).reindex(SEGS).values


def score_next(k):
    """Score the next k respondents with all four models."""
    end = min(N, ss.ls_i + k)
    if end <= ss.ls_i:
        return
    batch = live.iloc[ss.ls_i:end]
    p = predict_subscribe(M, batch)
    seg = predict_segment(M, batch)
    wtp = predict_wtp(M, batch)
    xy = pca_coords(M, batch)
    for j, (_, row) in enumerate(batch.iterrows()):
        ss.ls_clock += timedelta(seconds=random.randint(2, 11))
        off = next_offer(M, row.to_dict())
        ss.ls_rows.append({"Time": ss.ls_clock.strftime("%H:%M:%S"), "ID": row.respondent_id,
                           "Who": f"{'Self' if row.respondent_role == 'Myself' else 'Caregiver'} · {AGE_LABELS[int(row.patient_age_group)]}",
                           "P(subscribe)": float(p[j]), "Said (Q29)": int(row.purchase_intent),
                           "Correct": "✓" if int(p[j] >= 0.5) == int(row.interested) else "✗",
                           "ok": int(int(p[j] >= 0.5) == int(row.interested)), "seg": int(seg[j]), "Segment": SEGS[int(seg[j])],
                           "Spend AED/mo": float(wtp[j]), "Offer next": off["service"] if off else "Has all likely services",
                           "pc1": float(xy[j, 0]), "pc2": float(xy[j, 1])})
    ss.ls_i = end


def reset():
    ss.ls_i, ss.ls_rows, ss.ls_clock, ss.ls_running = 0, [], datetime.now(), True
    score_next(20)


if "ls_i" not in ss:
    reset()

st.title("📡 Live survey stream")
st.markdown("New respondents arrive one by one. Each one's questionnaire answers go through all four models the moment "
            "they arrive; their own Q29 answer then checks the subscribe prediction.")

c1, c2, c3 = st.columns([1, 1.4, 1])
if c1.button("⏸ Pause" if ss.ls_running else "▶ Run", use_container_width=True, disabled=ss.ls_i >= N):
    ss.ls_running = not ss.ls_running
    st.rerun()
speed = c2.selectbox("Speed", [1, 3, 10], index=1, format_func=lambda v: f"{v} respondent{'s' if v > 1 else ''} / sec",
                     label_visibility="collapsed")
if c3.button("↺ Restart", use_container_width=True):
    reset()
    st.rerun()


@st.fragment(run_every=1.0 if (ss.ls_running and ss.ls_i < N) else None)
def stream():
    if ss.ls_running and ss.ls_i < N:
        score_next(speed)
        if ss.ls_i >= N:
            ss.ls_running = False
            st.rerun()
    st.markdown(f"**{'🟢 Live' if ss.ls_running and ss.ls_i < N else '⏸ Paused' if ss.ls_i < N else '✅ Stream complete'}** · "
                f"{ss.ls_i} of {N} new respondents scored")
    R = pd.DataFrame(ss.ls_rows)
    n = len(R)
    acc = R.ok.mean()
    mix = np.bincount(R.seg, minlength=3) / n

    k = st.columns(6)
    k[0].metric("Respondents scored", f"{n}", f"of {N}", delta_color="off")
    k[1].metric("Predicted to subscribe", f"{R['P(subscribe)'].mean():.1%}", f"survey {M.survey.interested.mean():.1%}", delta_color="off")
    k[2].metric("Prediction accuracy (live)", f"{acc:.1%}", f"test set {TEST_ACC:.1%}", delta_color="off")
    k[3].metric("High-probable segment", f"{mix[0]:.1%}", f"survey {SURVEY_MIX[0]:.1%}", delta_color="off")
    k[4].metric("Avg predicted spend", f"{R['Spend AED/mo'].mean():.1f} AED")
    k[5].metric("Expected monthly revenue", f"{(R['P(subscribe)'] * R['Spend AED/mo']).sum():,.0f} AED",
                "Σ P(subscribe) × spend", delta_color="off")

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("**Latest 12 respondents**")
        feed = R.iloc[::-1].head(12)[["Time", "ID", "Who", "P(subscribe)", "Said (Q29)", "Correct", "Segment",
                                       "Spend AED/mo", "Offer next"]]
        st.dataframe(feed, hide_index=True, use_container_width=True, height=458, column_config={
            "P(subscribe)": st.column_config.ProgressColumn("P(subscribe)", format="%.2f", min_value=0, max_value=1),
            "Spend AED/mo": st.column_config.NumberColumn("Spend AED/mo", format="%.0f")})
    with right:
        st.markdown("**Segment mix: live vs survey**")
        fig = go.Figure([go.Bar(x=["Survey", "Live"], y=[SURVEY_MIX[i], mix[i]], name=s, marker_color=seg_color(s),
                                text=[f"{SURVEY_MIX[i]:.0%}", f"{mix[i]:.0%}"], textposition="inside",
                                hovertemplate="%{x}: %{y:.1%}<extra>" + s + "</extra>") for i, s in enumerate(SEGS)])
        fig.update_layout(barmode="stack")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 210), use_container_width=True, key="mix")
        st.markdown("**Predicted spend of live respondents**")
        fig = go.Figure(go.Histogram(x=R["Spend AED/mo"], xbins=dict(start=0, end=160, size=10), marker_color="#2a78d6"))
        fig.update_xaxes(title="AED / month")
        fig.update_yaxes(title="Respondents")
        st.plotly_chart(style(fig, 210, legend=False), use_container_width=True, key="whist")

    a, b, c = st.columns(3)
    with a:
        st.markdown("**1 · Classification: running accuracy**")
        run = R.ok.expanding().mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=run, mode="lines", name="Live accuracy",
                                 line=dict(color="#2a78d6", width=2), fill="tozeroy", fillcolor="rgba(42,120,214,0.08)"))
        fig.add_hline(y=TEST_ACC, line_dash="dash", line_color="#7a8491", annotation_text=f"Test set {TEST_ACC:.1%}")
        fig.update_xaxes(title="Respondents scored", range=[0, N])
        fig.update_yaxes(tickformat=".0%", range=[0.5, 1])
        st.plotly_chart(style(fig, 300, legend=False), use_container_width=True, key="acc")
    with b:
        st.markdown("**2 · Clustering: where new respondents land**")
        fig = go.Figure()
        S = M.survey
        for i, s in enumerate(SEGS):
            g = S[S.segment == s]
            fig.add_trace(go.Scattergl(x=g.pc1, y=g.pc2, mode="markers", name=s, legendgroup=s,
                                       marker=dict(size=4, color=seg_color(s), opacity=0.18), hoverinfo="skip"))
            lv = R[R.seg == i]
            fig.add_trace(go.Scattergl(x=lv.pc1, y=lv.pc2, mode="markers", name=f"{s} (live)", legendgroup=s, showlegend=False,
                                       marker=dict(size=9, color=seg_color(s), line=dict(color="white", width=1.5)),
                                       text=lv.ID, hovertemplate="%{text}<extra>" + s + "</extra>"))
        last = R.iloc[-1]
        fig.add_trace(go.Scatter(x=[last.pc1], y=[last.pc2], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=seg_color(int(last.seg)), width=2.5))))
        fig.update_xaxes(title="Component 1")
        fig.update_yaxes(title="Component 2")
        st.plotly_chart(style(fig, 300), use_container_width=True, key="pca")
    with c:
        st.markdown("**3 · Association: cross-sell offers made**")
        offers = R["Offer next"].value_counts()
        offers = offers[offers.index != "Has all likely services"]
        if len(offers):
            st.plotly_chart(hbar(offers.index.tolist(), offers.values.tolist(), height=300), use_container_width=True, key="offers")
        else:
            st.caption("No offers yet.")
    st.caption("4 · Regression: the spend column and histogram are gradient-boosting predictions of each respondent's Q31 answer.")


stream()
