"""Fill in the questionnaire and get all four model predictions instantly."""
import random

import streamlit as st

from src.config import SEGS, SERVICES
from src.models import next_offer, predict_segment, predict_subscribe, predict_wtp
from src.questionnaire import QUESTIONS, SECTIONS
from src.ui import get_models, page_setup, seg_color, sidebar_note

page_setup("Score a Respondent", "🧮")
sidebar_note()
M, live = get_models()
ss = st.session_state
SCORED = [q for q in QUESTIONS if q["col"] not in ("recommend_score", "wtp_monthly_aed", "purchase_channel", "payer",
                                                    "payment_method", "top_factor")]


def key(col):
    return f"q_{col}"


def load(idx):
    row = live.iloc[idx]
    for q in SCORED:
        if q["kind"] == "multi":
            for col, _ in q["options"]:
                ss[key(col)] = bool(int(row[col]))
        elif q["kind"] == "num":
            ss[key(q["col"])] = int(row[q["col"]])
        elif q["kind"] == "cat":
            ss[key(q["col"])] = str(row[q["col"]])
        else:
            ss[key(q["col"])] = int(row[q["col"]])
    ss.score_src, ss.score_idx = row.respondent_id, idx


# Streamlit clears widget values after you visit another page, so reload the respondent if any are missing
if "score_src" not in ss or any(key(q["col"]) not in ss for q in SCORED if q["kind"] != "multi"):
    load(ss.get("score_idx", 1))

st.title("🧮 Score a respondent")
st.markdown("Change any answer and all four models re-score instantly.")
b1, b2 = st.columns([1, 4])
if b1.button("Load a random live respondent", use_container_width=True):
    load(random.randrange(len(live)))
    st.rerun()
b2.caption(f"Started from live respondent {ss.score_src}.")

results = st.container()

for code, title, _ in SECTIONS:
    qs = [q for q in SCORED if q["sec"] == code]
    if not qs:
        continue
    with st.expander(f"Section {code} – {title}", expanded=code in ("C", "E")):
        cols = st.columns(3)
        i = 0
        for q in qs:
            label = f"{q['id']} {q['text']}"
            if q["kind"] == "multi":
                st.markdown(f"**{label}**")
                mc = st.columns(3)
                for j, (col, lbl) in enumerate(q["options"]):
                    mc[j % 3].checkbox(lbl, key=key(col))
                continue
            target = cols[i % 3]
            i += 1
            if q["kind"] == "num":
                target.number_input(label, min_value=0, max_value=5000, step=5, key=key(q["col"]))
            else:
                values = [v for v, _ in q["options"]]
                labels = dict(q["options"])
                target.selectbox(label, values, format_func=lambda v, labels=labels: labels[v], key=key(q["col"]))

row = {}
for q in SCORED:
    if q["kind"] == "multi":
        for col, _ in q["options"]:
            row[col] = int(ss[key(col)])
    else:
        row[q["col"]] = ss[key(q["col"])]

p = float(predict_subscribe(M, row)[0])
seg = int(predict_segment(M, row)[0])
wtp = float(predict_wtp(M, row)[0])
off = next_offer(M, row)
with results:
    r = st.columns(4)
    r[0].metric("1 · P(subscribe)", f"{p:.1%}", "Predicted to subscribe" if p >= 0.5 else "Predicted not to subscribe",
                delta_color="normal" if p >= 0.5 else "inverse")
    r[1].markdown(f"**2 · Segment**<br><span style='font-size:1.6rem;font-weight:700;color:{seg_color(seg)}'>{SEGS[seg]}</span>",
                  unsafe_allow_html=True)
    r[2].metric("4 · Predicted spend", f"{wtp:.0f} AED/month")
    if off:
        r[3].metric("3 · Offer next", off["service"], f"{off['confidence']:.0%} of people wanting {' + '.join(off['because'])} want it",
                    delta_color="off")
    else:
        r[3].metric("3 · Offer next", "No rule applies", "Tick more services in Section E", delta_color="off")
    st.divider()
