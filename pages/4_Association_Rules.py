import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import apriori
from src.config import SERVICES, SVC_LABEL
from src.ui import get_models, hbar, page_setup, sidebar_note, style

page_setup("Association Rules", "🔗")
sidebar_note()
M, _ = get_models()
df = M.survey
basket = df[SERVICES]

st.title("🔗 Association rules: which services are wanted together?")
st.markdown("Apriori over each respondent's Section E basket (Yes = wanted). A rule *A + B → C* reads: people who "
            "want A and B also want C. **Support** = share of all respondents wanting all three; **confidence** = "
            "share of A+B wanters who also want C; **lift** = how many times likelier than average.")

c1, c2, c3, c4 = st.columns(4)
min_sup = c1.slider("Minimum support", 0.05, 0.40, 0.15, 0.01)
min_conf = c2.slider("Minimum confidence", 0.30, 0.95, 0.60, 0.05)
min_lift = c3.slider("Minimum lift", 1.00, 2.00, 1.15, 0.05)
size = c4.radio("Rule shape", ["2 services → 3rd", "1 → 1", "Any"], horizontal=False)


@st.cache_data
def rules_for(min_sup):
    freq = apriori.frequent_itemsets(basket, min_sup)
    return freq, apriori.association_rules(freq, len(basket), min_confidence=0.3, min_lift=1.0)


freq, rules = rules_for(round(min_sup, 2))
r = rules[(rules.confidence >= min_conf) & (rules.lift >= min_lift)]
if size == "2 services → 3rd":
    r = r[r.n_antecedent == 2]
elif size == "1 → 1":
    r = r[r.n_antecedent == 1]

k1, k2, k3 = st.columns(3)
k1.metric("Frequent itemsets", f"{len(freq)}")
k2.metric("Rules shown", f"{len(r)}")
k3.metric("Average basket", f"{basket.sum(axis=1).mean():.1f} of 10 services")

show = pd.DataFrame({
    "If they want…": [" + ".join(SVC_LABEL[a] for a in x) for x in r.antecedent],
    "…they also want": [SVC_LABEL[x[0]] for x in r.consequent],
    "Support": r.support.values, "Confidence": r.confidence.values, "Lift": r.lift.values, "People": r["count"].values})
st.dataframe(show, hide_index=True, use_container_width=True, height=380,
             column_config={"Support": st.column_config.ProgressColumn("Support", format="%.2f", min_value=0, max_value=1),
                            "Confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
                            "Lift": st.column_config.NumberColumn("Lift", format="%.2f×")})

st.subheader("Cross-sell finder")
pick = st.multiselect("A customer already wants…", [SVC_LABEL[s] for s in SERVICES],
                      default=["Appointment reminders", "Lab test reminders"])
inv = {v: k for k, v in SVC_LABEL.items()}
have = {inv[p] for p in pick}
if have:
    cand = rules[(rules.confidence >= min_conf) & (rules.lift >= min_lift)]
    cand = cand[[set(a) <= have and c[0] not in have for a, c in zip(cand.antecedent, cand.consequent)]]
    if len(cand):
        best = (cand.assign(service=[SVC_LABEL[c[0]] for c in cand.consequent],
                            because=[" + ".join(SVC_LABEL[a] for a in x) for x in cand.antecedent])
                .sort_values(["confidence", "lift"], ascending=False).drop_duplicates("service"))
        top = best.iloc[0]
        st.success(f"Offer **{top.service}** next: {top.confidence:.0%} of people who want {top.because} also want it "
                   f"(lift {top.lift:.2f}×).")
        st.dataframe(best[["service", "because", "confidence", "lift", "support"]].rename(columns={
            "service": "Offer", "because": "Because they want", "confidence": "Confidence", "lift": "Lift",
            "support": "Support"}).round(3), hide_index=True, use_container_width=True)
    else:
        st.info("No rule meets the thresholds for this basket. Lower the thresholds or pick different services.")

left, right = st.columns([1.2, 1])
with left:
    st.markdown("**Pairwise lift between services** (blue: wanted together more than chance)")
    L = M.lift.copy()
    labels = [SVC_LABEL[s] for s in SERVICES]
    z = L.values.astype(float)
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale="RdBu", zmid=1, zmin=0.4, zmax=2.0,
                               text=np.where(np.isnan(z), "", np.round(z, 2).astype(str)), texttemplate="%{text}",
                               hovertemplate="%{y} + %{x}<br>Lift %{z:.2f}×<extra></extra>", colorbar=dict(title="Lift")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 520, legend=False), use_container_width=True)
with right:
    st.markdown("**Share of respondents wanting each service**")
    sup = basket.mean().sort_values(ascending=False)
    st.plotly_chart(hbar([SVC_LABEL[s] for s in sup.index], sup.values.round(3).tolist(),
                         text=[f"{v:.0%}" for v in sup.values]), use_container_width=True)
