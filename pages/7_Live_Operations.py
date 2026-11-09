"""Live simulation of the MediConnect service running hour by hour, with costs and profit."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AGE_LABELS, CHANNELS, SEGS, SERVICES, SVC_LABEL
from src.ops_sim import Costs, OpsSim, Scenario
from src.ui import get_pool, hbar, money, page_setup, seg_color, sidebar_note, style

page_setup("Live Operations", "⚙️")
pool = get_pool()
ss = st.session_state
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SPEEDS = {"1 sim hour / sec": 1, "6 sim hours / sec": 6, "1 sim day / sec": 24, "1 sim week / sec": 168}

# ---------------- scenario and cost levers (sidebar)
st.sidebar.header("Scenario")
price = st.sidebar.slider("Monthly price (AED)", 15, 99, 39)
leads = st.sidebar.slider("New leads per day", 10, 250, 70, 5, help="People who reach the WhatsApp sign-up")
effect = st.sidebar.slider("Reminder effect", 0, 80, 45, 5, format="%d%%",
                           help="Share of would-be missed doses that a reminder prevents")
followup = st.sidebar.toggle("30-min follow-up nudge", True)
cg = st.sidebar.toggle("Caregiver alerts", True)
with st.sidebar.expander("Unit costs (AED)"):
    fixed = st.number_input("Fixed costs per month", 0, 1_000_000, 75000, 5000)
    per_lead = st.number_input("Cost per lead", 0.0, 500.0, 20.0, 1.0)
    wa = st.number_input("WhatsApp message", 0.0, 1.0, 0.069, 0.005, format="%.3f",
                         help="Meta UAE utility rate AED 0.0576 + 20% provider markup")
    sms = st.number_input("SMS", 0.0, 1.0, 0.10, 0.01, format="%.2f")
    call = st.number_input("Automated call", 0.0, 5.0, 0.30, 0.05, format="%.2f")
    push = st.number_input("App notification", 0.0, 1.0, 0.005, 0.005, format="%.3f")
    pay_pct = st.number_input("Payment fee %", 0.0, 10.0, 2.9, 0.1) / 100
    support = st.number_input("Support per subscriber / month", 0.0, 50.0, 3.0, 0.5)
sidebar_note()


def new_sim():
    sim = OpsSim(pool, Scenario(), Costs())
    apply_levers(sim)
    sim.run(24 * 21 + 7)   # warm up three simulated weeks so every chart has history
    return sim


def apply_levers(sim):
    sim.sc.price, sim.sc.leads_per_day, sim.sc.reminder_effect = price, leads, effect / 100
    sim.sc.followup, sim.sc.caregiver_alerts = followup, cg
    c = sim.cost
    c.fixed_month, c.per_lead, c.whatsapp, c.sms, c.call, c.push = fixed, per_lead, wa, sms, call, push
    c.payment_pct, c.support_month = pay_pct, support


if "ops" not in ss:
    ss.ops, ss.ops_running = new_sim(), True
apply_levers(ss.ops)

st.title("⚙️ MediConnect live operations")
st.markdown("The service running hour by hour. Every simulated patient is a survey respondent, so their dose times, "
            "channel, language, services and willingness to pay come from their answers.")
c1, c2, c3 = st.columns([1, 1.6, 1])
if c1.button("⏸ Pause" if ss.ops_running else "▶ Run", use_container_width=True):
    ss.ops_running = not ss.ops_running
    st.rerun()
speed = SPEEDS[c2.selectbox("Speed", list(SPEEDS), index=1, label_visibility="collapsed")]
if c3.button("↺ Restart", use_container_width=True):
    ss.ops = new_sim()
    st.rerun()


@st.fragment(run_every=1.0 if ss.ops_running else None)
def live():
    sim: OpsSim = ss.ops
    if ss.ops_running:
        sim.run(speed)
    d, h = sim.day, sim.hour
    st.markdown(f"### {'🟢 Running' if ss.ops_running else '⏸ Paused'} · Day {d} · {h:02d}:00 · {WEEKDAYS[(d - 1) % 7]}")
    E = sim.economics()
    adh, base = sim.seven_day_adherence()
    T = sim.today
    k = st.columns(6)
    k[0].metric("Active subscribers", f"{len(sim.patients):,}", f"{sim.c['subs']:,} joined · {sim.c['churned']:,} cancelled", delta_color="off")
    k[1].metric("Monthly recurring revenue", f"{len(sim.patients) * sim.sc.price:,.0f} AED")
    if adh is not None:
        k[2].metric("Dose adherence, 7 days", f"{adh:.1%}", f"{(adh - base) * 100:+.1f} pts vs {base:.1%} before")
    k[3].metric("Doses rescued today", f"{T['late'] + T['rescued']}", "by follow-up or caregiver", delta_color="off")
    if E:
        k[4].metric(f"Profit, last {E['days']} days", f"{money(E['profit'])} AED",
                    f"revenue {E['rev']:,.0f} − costs {E['cost']:,.0f}", delta_color="off")
    k[5].metric("Cash position", f"{money(sim.cash)} AED",
                f"lowest {money(sim.cash_min)} on day {sim.cash_min_day}" if sim.cash_min < 0 else "never below zero", delta_color="off")

    D = sim.daily_df()
    tabs = st.tabs(["👥 Patients & messages", "💊 Adherence", "📈 Growth", "💵 Costs & profit", "🩺 Services"])

    with tabs[0]:
        left, right = st.columns([1.5, 1])
        icon = {"taken": "✅", "late": "☑️", "rescued": "🟨", "missed": "❌", "no medicine": "🚫", "due": "⬜"}
        rows = []
        for p in sim.tracked:
            recent = p.hist[-7:]
            rows.append({"Patient": p.pid, "Segment": SEGS[p.seg].replace(" probable", ""),
                         "Who": f"{'Caregiver' if p.care else 'Self'} · {AGE_LABELS[p.age]}", "Channel": CHANNELS[p.ch],
                         "Today's doses": " ".join(f"{icon[s]}{hh:02d}" for s, hh in zip(p.slots, p.hours)),
                         "7-day adherence": f"{sum(recent) / len(recent):.0%}" if recent else "new", "Supply (days)": max(p.supply, 0)})
        with left:
            st.markdown("**Followed patients: today's doses**  \n✅ taken · ☑️ after follow-up · 🟨 after caregiver alert · ❌ missed · 🚫 no medicine · ⬜ due")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with right:
            st.markdown("**Live message log**")
            st.dataframe(pd.DataFrame(list(sim.log)[:18]), hide_index=True, use_container_width=True, height=420)

    with tabs[1]:
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Daily dose adherence**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=D.day, y=D.baseline, name="Same patients before (survey Q16)",
                                     line=dict(color="#7a8491", dash="dash")))
            fig.add_trace(go.Scatter(x=D.day, y=D.adherence, name="With MediConnect", line=dict(color="#2a78d6", width=2)))
            fig.update_yaxes(tickformat=".0%", range=[0.6, 1])
            fig.update_xaxes(title="Day")
            st.plotly_chart(style(fig, 330), use_container_width=True, key="adh")
        with b:
            st.markdown("**Today's doses by outcome**")
            out = [("Taken on first reminder", T["taken"], "#0ca30c"), ("Taken after follow-up", T["late"], "#0ca30c"),
                   ("Confirmed by caregiver", T["rescued"], "#fab219"), ("Missed", T["missed"] - T["nomed"], "#d03b3b"),
                   ("Missed: no medicine", T["nomed"], "#d03b3b")]
            tot = max(1, T["doses"])
            st.plotly_chart(hbar([o[0] for o in out], [o[1] for o in out], text=[f"{o[1]} · {o[1] / tot:.0%}" for o in out],
                                 colors=[o[2] for o in out], height=330), use_container_width=True, key="outcomes")
        with c:
            st.markdown("**Messages sent today by channel** (from Q25)")
            st.plotly_chart(hbar(CHANNELS, T["msgs"], text=[str(v) for v in T["msgs"]], height=330),
                            use_container_width=True, key="channels")

    with tabs[2]:
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Active subscribers**")
            fig = go.Figure(go.Scatter(x=D.day, y=D.active, fill="tozeroy", line=dict(color="#2a78d6", width=2)))
            fig.update_xaxes(title="Day")
            st.plotly_chart(style(fig, 320, legend=False), use_container_width=True, key="subs")
        with b:
            st.markdown("**Monthly recurring revenue, AED**")
            fig = go.Figure(go.Scatter(x=D.day, y=D.mrr, fill="tozeroy", line=dict(color="#2a78d6", width=2)))
            fig.update_xaxes(title="Day")
            st.plotly_chart(style(fig, 320, legend=False), use_container_width=True, key="mrr")
        with c:
            st.markdown("**Sign-up funnel, all days**  \nleads → said 4–5 on Q29 → price within Q31 budget → onboarded")
            f = [("Leads", sim.c["leads"]), ("Likely to subscribe", sim.c["interested"]),
                 ("Price within budget", sim.c["afford"]), ("Subscribed", sim.c["subs"])]
            L = max(1, sim.c["leads"])
            st.plotly_chart(hbar([x[0] for x in f], [x[1] for x in f], text=[f"{x[1]:,} · {x[1] / L:.0%}" for x in f], height=300),
                            use_container_width=True, key="funnel")

    with tabs[3]:
        if E:
            cover_ok = len(sim.patients) >= E["needed"]
            m = st.columns(5)
            m[0].metric("Break-even subscribers", f"{len(sim.patients):,} / {E['needed']:,.0f}" if E["needed"] != float("inf") else "Never",
                        "profitable" + (f" since day {sim.profit_day}" if sim.profit_day else "") if cover_ok else
                        f"{max(0, E['needed'] - len(sim.patients)):,.0f} more to cover fixed + lead costs", delta_color="off")
            m[1].metric("Margin per subscriber", f"{E['contrib']:.1f} AED/mo", "after messages, payments, support", delta_color="off")
            m[2].metric("Cost to win a subscriber", f"{E['cac']:.0f} AED" if E["cac"] else "–",
                        f"{per_lead:.0f} AED/lead ÷ {E['conv']:.0%} convert" if E["conv"] else None, delta_color="off")
            m[3].metric("Customer value : cost to win", f"{E['ltv_cac']:.1f}×" if E["ltv_cac"] else "–",
                        f"churn {E['churn_mo']:.1%}/month · 3× is healthy" if E["churn_mo"] else None, delta_color="off")
            m[4].metric("Payback", f"{E['payback']:.1f} months" if E["payback"] else "–")
            if E["needed"] != float("inf"):
                st.progress(min(1.0, len(sim.patients) / E["needed"]),
                            text=f"Subscribers vs break-even (covers AED {E['cover_month']:,.0f}/month of fixed and lead costs)")
            a, b, c = st.columns(3)
            with a:
                st.markdown("**Daily revenue vs costs** (AED/day, 7-day average)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=D.day, y=D.rev7, name="Revenue", line=dict(color="#0ca30c", width=2)))
                fig.add_trace(go.Scatter(x=D.day, y=D.cost7, name="Total costs", line=dict(color="#d03b3b", width=2)))
                fig.update_xaxes(title="Day")
                st.plotly_chart(style(fig, 330), use_container_width=True, key="pnl")
            with b:
                st.markdown(f"**Cash position** (funding needed: AED {max(0, -sim.cash_min):,.0f})")
                col = "#0ca30c" if sim.cash >= 0 else "#d03b3b"
                fig = go.Figure(go.Scatter(x=D.day, y=D.cash, fill="tozeroy", line=dict(color=col, width=2)))
                fig.add_hline(y=0, line_color="#7a8491")
                fig.update_xaxes(title="Day")
                st.plotly_chart(style(fig, 330, legend=False), use_container_width=True, key="cash")
            with c:
                st.markdown(f"**Where the money goes** (last {E['days']} days)")
                bd = sorted(E["breakdown"].items(), key=lambda kv: -kv[1])
                tc = max(1, E["cost"])
                st.plotly_chart(hbar([x[0] for x in bd], [round(x[1]) for x in bd], text=[f"{x[1]:,.0f} · {x[1] / tc:.0%}" for x in bd],
                                     colors="#d03b3b", height=330), use_container_width=True, key="costs")
            per = pd.DataFrame({"Per subscriber per month": list(E["per_sub"]) + ["Margin"],
                                "AED": [round(v, 2) for v in E["per_sub"].values()] + [round(E["contrib"], 2)]})
            st.dataframe(per, hide_index=True)

    with tabs[4]:
        a, b, c = st.columns(3)
        C = sim.c
        with a:
            st.markdown("**Care events, all days**")
            ev = [("Refill alerts sent", C["refill_alerts"]), ("Pharmacy deliveries", C["orders"]), ("Medicine run-outs", C["run_outs"]),
                  ("Doctor appointments", C["appts"]), ("Lab tests due", C["labs"]), ("Caregiver alerts", C["cg_alerts"]),
                  ("Teleconsults booked", C["tele"])]
            txt = [f"{v:,}" for _, v in ev]
            txt[3] += f" · {C['appt_attended'] / max(1, C['appts']):.0%} attended"
            txt[4] += f" · {C['labs_done'] / max(1, C['labs']):.0%} done"
            st.plotly_chart(hbar([e[0] for e in ev], [e[1] for e in ev], text=txt, height=330), use_container_width=True, key="care")
        with b:
            st.markdown("**Subscribers by segment**")
            cnt = [sum(1 for p in sim.patients if p.seg == i) for i in range(3)]
            tot = max(1, len(sim.patients))
            st.plotly_chart(hbar(SEGS, cnt, text=[f"{v:,} · {v / tot:.0%}" for v in cnt], colors=[seg_color(i) for i in range(3)], height=240),
                            use_container_width=True, key="segs")
        with c:
            st.markdown("**Services active subscribers use**")
            use = [(SVC_LABEL[s], sum(1 for p in sim.patients if (p.svc >> i) & 1)) for i, s in enumerate(SERVICES)]
            use.sort(key=lambda x: -x[1])
            st.plotly_chart(hbar([u[0] for u in use], [u[1] for u in use], text=[f"{u[1] / tot:.0%}" for u in use], height=360),
                            use_container_width=True, key="svcs")


live()
st.caption("Assumptions: 62% of qualified leads finish onboarding; a follow-up rescues 30% of unconfirmed doses and a caregiver "
           "alert 40%; pharmacy delivery earns AED 10 per order and teleconsult booking AED 15. Churn rises when the price "
           "exceeds the patient's Q31 answer and when adherence falls. All figures are simulated.")
