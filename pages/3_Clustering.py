import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CLU, SEG_COLORS, pretty
from src.models import cluster_any_k, segment_profiles
from src.ui import get_models, page_setup, sidebar_note, style

page_setup("Clustering", "🧩")
sidebar_note()
M, _ = get_models()
df = M.survey.copy()

st.title("🧩 Clustering: most, medium and least probable customers")
st.markdown("K-means on nine pain and readiness answers (Q11, Q16–Q18, Q20, Q21, Q23, Q24, Q27), standardised. "
            "Clusters are then ranked by how many of their members said they'd subscribe (Q29 = 4–5).")

k = st.select_slider("Number of segments (k)", options=[2, 3, 4, 5, 6], value=3,
                     help="k = 3 gives the High / Medium / Low probable tiers")
labels, names, km, name_of = cluster_any_k(M, k)
df["seg_k"] = labels
palette = list(SEG_COLORS.values()) + ["#e87ba4", "#6250d6", "#eda100"]
colors = {n: palette[i] for i, n in enumerate(names)}

prof = segment_profiles(df, "seg_k", names)
cols = st.columns(len(prof))
for col, (_, r) in zip(cols, prof.iterrows()):
    col.metric(r["Segment"], f"{r['Would subscribe (Q29 4–5)']:.0%} subscribe",
               f"{r['Respondents']} people · {r['Avg spend, AED/mo']:.0f} AED/mo", delta_color="off")

left, right = st.columns([1.3, 1])
with left:
    st.markdown("**Respondents on the two main components** (PCA of the nine inputs)")
    fig = go.Figure()
    for n in names:
        g = df[df.seg_k == n]
        fig.add_trace(go.Scattergl(x=g.pc1, y=g.pc2, mode="markers", name=n,
                                   marker=dict(size=6, color=colors[n], opacity=0.55),
                                   customdata=np.stack([g.respondent_id, g.purchase_intent, g.wtp_monthly_aed], axis=1),
                                   hovertemplate="%{customdata[0]}<br>Q29: %{customdata[1]}<br>Spend: %{customdata[2]} AED<extra>" + n + "</extra>"))
    ev = M.pca.explained_variance_ratio_
    fig.update_xaxes(title=f"Component 1 ({ev[0]:.0%} of variance)")
    fig.update_yaxes(title=f"Component 2 ({ev[1]:.0%})")
    st.plotly_chart(style(fig, 460), use_container_width=True)
with right:
    st.markdown("**Choosing k**")
    kk = list(M.silhouette.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=kk, y=[M.silhouette[i] for i in kk], name="Silhouette (higher = better separated)",
                         marker_color=["#2a78d6" if i == k else "#b7d3f6" for i in kk]))
    fig.update_xaxes(title="k", dtick=1)
    fig.update_yaxes(title="Silhouette score")
    st.plotly_chart(style(fig, 220, legend=False), use_container_width=True)
    fig = go.Figure(go.Scatter(x=kk, y=[M.inertia[i] for i in kk], mode="lines+markers", line=dict(color="#2a78d6")))
    fig.update_xaxes(title="k (elbow method)", dtick=1)
    fig.update_yaxes(title="Within-cluster inertia")
    st.plotly_chart(style(fig, 220, legend=False), use_container_width=True)
    st.caption("Silhouette scores are moderate: survey segments shade into each other rather than forming islands. "
               "k = 3 is chosen because it answers the three-tier business question.")

st.markdown("**Segment profiles**")
pct_cols = ["Share", "Would subscribe (Q29 4–5)", "Patient aged 60+", "Caregiver respondents"]
shown = prof.copy()
for c in shown.columns:
    if c in pct_cols:
        shown[c] = shown[c].map(lambda v: f"{v:.0%}")
    elif shown[c].dtype.kind == "f":
        shown[c] = shown[c].map(lambda v: f"{v:.2f}")
    else:
        shown[c] = shown[c].astype(str)
st.dataframe(shown.set_index("Segment").T, use_container_width=True)

st.markdown("**What each segment looks like** (average of each input, in standard deviations from the survey mean)")
centres = pd.DataFrame(km.cluster_centers_, columns=[pretty(c) for c in CLU])
centres.index = [name_of[c] for c in range(k)]
centres = centres.loc[[n for n in names if n in centres.index]]
fig = go.Figure(go.Heatmap(z=centres.values, x=centres.columns, y=centres.index, colorscale="RdBu", zmid=0,
                           text=np.round(centres.values, 2), texttemplate="%{text}", colorbar=dict(title="SD")))
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style(fig, 120 + 50 * k, legend=False), use_container_width=True)
