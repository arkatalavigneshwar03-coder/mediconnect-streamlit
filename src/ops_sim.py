"""
Live operations simulation of the MediConnect service, one simulated hour per tick.

Every simulated patient is drawn from a real survey respondent, so dose times (Q12), channel (Q25),
language (Q26), services (Section E), willingness to pay (Q31) and baseline adherence (Q16) come from the data.

Per hour: leads arrive 09:00-21:59 and convert if Q29 >= 4 and Q31 covers the price; reminders go out at each
patient's dose times; unconfirmed doses get a 30-minute follow-up and then a caregiver alert. Per day: refills,
appointments, lab tests, teleconsults, churn, billing and costs.
"""
import random
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .config import AGE_LABELS, CHANNELS, LANGS, SEGS, SERVICES

DOSE_HOURS = {1: [9], 2: [8, 20], 3: [8, 14, 20], 4: [8, 12, 16, 21]}
MISSED_PER_WEEK = {0: 0.3, 1: 1.5, 2: 4, 3: 7}
SVC = {k: i for i, k in enumerate(["med", "appt", "lab", "refill", "scan", "care", "deliv", "vitals", "tele", "report"])}
VISITS = [0.5, 1.5, 4, 8]
LABS = [0.3, 1.5, 4, 8]


@dataclass
class Scenario:
    price: float = 39.0             # AED per month
    leads_per_day: int = 70
    reminder_effect: float = 0.45   # share of would-be missed doses a reminder prevents
    followup: bool = True
    caregiver_alerts: bool = True
    onboarding_rate: float = 0.62   # share of qualified leads who finish WhatsApp onboarding
    followup_recovery: float = 0.30
    caregiver_recovery: float = 0.40
    delivery_fee: float = 10.0      # AED commission per pharmacy delivery
    tele_fee: float = 15.0          # AED per teleconsult booking


@dataclass
class Costs:
    fixed_month: float = 75000.0
    per_lead: float = 20.0
    whatsapp: float = 0.069         # Meta UAE utility AED 0.0576 + 20% provider markup
    sms: float = 0.10
    call: float = 0.30
    push: float = 0.005
    payment_pct: float = 0.029
    payment_fixed: float = 1.0      # AED per monthly charge
    support_month: float = 3.0      # AED per subscriber per month

    def per_channel(self):
        return [self.whatsapp, self.whatsapp, self.sms, self.push, self.call]   # order = CHANNELS


@dataclass
class Patient:
    pid: str
    seg: int
    wtp: float
    doses: int
    base: float
    svc: int
    ch: int
    lang: int
    care: int
    age: int
    visits: float
    labs: float
    supply: int
    joined: int
    gap: int = 0
    ordered: bool = False
    slots: list = field(default_factory=list)
    hist: list = field(default_factory=list)

    @property
    def hours(self):
        return DOSE_HOURS[self.doses]

    def has(self, key):
        return (self.svc >> SVC[key]) & 1


def build_pool(survey_with_segments: pd.DataFrame) -> list:
    pool = []
    for r in survey_with_segments.itertuples(index=False):
        d = r._asdict()
        doses_week = d["doses_per_day"] * 7
        base = max(0.35, 1 - MISSED_PER_WEEK[d["missed_doses_week"]] / doses_week)
        mask = sum(1 << i for i, s in enumerate(SERVICES) if int(d[s]) == 1)
        pool.append({"seg": SEGS.index(d["segment"]), "intent": int(d["purchase_intent"]), "wtp": float(d["wtp_monthly_aed"]),
                     "doses": int(d["doses_per_day"]), "base": base, "svc": mask,
                     "ch": CHANNELS.index(d["preferred_channel"]), "lang": LANGS.index(d["preferred_language"]),
                     "care": int(d["respondent_role"] != "Myself"), "age": int(d["patient_age_group"]),
                     "visits": VISITS[int(d["doctor_visits"])], "labs": LABS[int(d["lab_tests"])]})
    return pool


def _new_today():
    return {"doses": 0, "taken": 0, "late": 0, "rescued": 0, "missed": 0, "nomed": 0, "base_sum": 0.0,
            "msgs": [0] * 5, "new_subs": 0, "churn": 0, "leads": 0, "rev": 0.0,
            "c_msg": [0.0] * 5, "c_acq": 0.0, "c_pay": 0.0, "c_sup": 0.0, "c_fix": 0.0}


class OpsSim:
    def __init__(self, pool, scenario: Scenario = None, costs: Costs = None, seed=None):
        self.pool = pool
        self.sc = scenario or Scenario()
        self.cost = costs or Costs()
        self.rng = random.Random(seed)
        self.t = 0
        self.patients = []
        self.next_id = 1
        self.c = dict(leads=0, interested=0, afford=0, subs=0, churned=0, revenue=0.0, orders=0, refill_alerts=0,
                      run_outs=0, appts=0, appt_attended=0, appt_rem=0, labs=0, labs_done=0, tele=0, cg_alerts=0)
        self.deliveries = []
        self.daily = []
        self.tracked = []
        self.log = deque(maxlen=60)
        self.cash = 0.0
        self.cash_min = 0.0
        self.cash_min_day = 0
        self.profit_day = None
        self.today = _new_today()
        self._minute = 0
        self._minute_t = -1

    # ----- time helpers
    @property
    def day(self):
        return self.t // 24 + 1

    @property
    def hour(self):
        return self.t % 24

    def _stamp(self):
        if self._minute_t != self.t:
            self._minute_t, self._minute = self.t, 1
        self._minute = min(59, self._minute + 1 + self.rng.randrange(4))
        return f"D{self.day} {self.hour:02d}:{self._minute:02d}"

    def _log(self, icon, text):
        self.log.appendleft({"time": self._stamp(), "event": icon, "detail": text})

    def _send(self, ch):
        self.today["msgs"][ch] += 1
        self.today["c_msg"][ch] += self.cost.per_channel()[ch]

    # ----- simulation
    def run(self, hours: int):
        for _ in range(hours):
            self.tick()

    def tick(self):
        h = self.hour
        if h == 0 and self.t > 0:
            self._close_day()
        if h == 0:
            self._start_day()
        if 9 <= h <= 21:
            expected = self.sc.leads_per_day / 13
            k = int(expected) + (1 if self.rng.random() < expected % 1 else 0)
            for _ in range(k):
                self._acquire()
        for p in self.patients:
            if h in p.hours:
                self._dose(p, p.hours.index(h))
        if h == 15:
            keep = []
            for d in self.deliveries:
                if d["day"] > self.day:
                    keep.append(d)
                    continue
                p = d["p"]
                p.supply, p.gap, p.ordered = 30, 0, False
                if p in self.tracked:
                    self._log("▣", f"Refill delivered to {p.pid}")
            self.deliveries = keep
        self.t += 1

    def _acquire(self):
        sc, rng, T = self.sc, self.rng, self.today
        self.c["leads"] += 1
        T["leads"] += 1
        T["c_acq"] += self.cost.per_lead
        r = self.pool[rng.randrange(len(self.pool))]
        if not (r["intent"] >= 4 or (r["intent"] == 3 and rng.random() < 0.08)):
            return
        self.c["interested"] += 1
        if not (r["wtp"] >= sc.price or (r["wtp"] >= 0.8 * sc.price and rng.random() < 0.3)):
            return
        self.c["afford"] += 1
        if rng.random() > sc.onboarding_rate:
            return
        p = Patient(pid=f"P{self.next_id:04d}", seg=r["seg"], wtp=r["wtp"], doses=r["doses"], base=r["base"],
                    svc=r["svc"], ch=r["ch"], lang=r["lang"], care=r["care"], age=r["age"], visits=r["visits"],
                    labs=r["labs"], supply=1 + rng.randrange(30), joined=self.day)
        p.slots = ["due"] * len(p.hours)
        self.next_id += 1
        self.patients.append(p)
        self.c["subs"] += 1
        T["new_subs"] += 1
        if len(self.tracked) < 8:
            self.tracked.append(p)
        if p in self.tracked or rng.random() < 0.25:
            who = "caregiver for a" if p.care else "patient"
            self._log("+", f"New subscriber {p.pid} · {who} {AGE_LABELS[p.age]} · {p.doses} dose(s)/day · "
                           f"{CHANNELS[p.ch]}, {LANGS[p.lang]}")

    def _dose(self, p: Patient, si: int):
        sc, rng, T = self.sc, self.rng, self.today
        T["doses"] += 1
        T["base_sum"] += p.base
        tr = p in self.tracked
        self._send(p.ch)
        if p.supply <= 0:
            p.slots[si] = "no medicine"
            T["missed"] += 1
            T["nomed"] += 1
            if tr:
                self._log("✗", f"{p.pid} has no medicine left; dose missed")
            return
        p1 = p.base + sc.reminder_effect * (1 - p.base)
        if rng.random() < p1:
            p.slots[si] = "taken"
            T["taken"] += 1
            if tr:
                self._log("✓", f'{p.pid} replied "Taken" to the {CHANNELS[p.ch]} reminder')
            return
        if sc.followup:
            self._send(p.ch)
            if rng.random() < sc.followup_recovery:
                p.slots[si] = "late"
                T["late"] += 1
                if tr:
                    self._log("✓", f"{p.pid} took the dose after the 30-minute follow-up")
                return
        if sc.caregiver_alerts and p.has("care"):
            self.c["cg_alerts"] += 1
            self._send(0)
            ok = rng.random() < sc.caregiver_recovery
            if tr or rng.random() < 0.15:
                self._log("!", f"Caregiver alert for {p.pid}: dose unconfirmed"
                               f"{'; family confirmed it was taken' if ok else '; no confirmation'}")
            if ok:
                p.slots[si] = "rescued"
                T["rescued"] += 1
                return
        p.slots[si] = "missed"
        T["missed"] += 1
        if tr:
            self._log("✗", f"{p.pid} missed the {p.hours[si]:02d}:00 dose")

    def _start_day(self):
        sc, rng, T, d = self.sc, self.rng, self.today, self.day
        T["c_fix"] += self.cost.fixed_month / 30
        keep = []
        for p in self.patients:
            p.slots = ["due"] * len(p.hours)
            recent = p.hist[-7:]
            adh = sum(recent) / len(recent) if recent else 0.9
            hz = 0.0018 + 0.006 * max(0.0, (sc.price - p.wtp) / sc.price) + 0.006 * max(0.0, 0.8 - adh)
            if d - p.joined > 2 and rng.random() < hz:
                self.c["churned"] += 1
                T["churn"] += 1
                self._log("−", f"{p.pid} cancelled after {d - p.joined} days")
                continue
            keep.append(p)
            tr = p in self.tracked
            if p.gap > 0:
                p.gap -= 1
                if p.gap == 0:
                    p.supply = 30
            elif p.supply > 0:
                p.supply -= 1
                if p.supply == 5 and p.has("refill"):
                    self.c["refill_alerts"] += 1
                    self._send(p.ch)
                    if p.has("deliv") and rng.random() < 0.78:
                        self.c["orders"] += 1
                        self.c["revenue"] += sc.delivery_fee
                        T["rev"] += sc.delivery_fee
                        p.ordered = True
                        self.deliveries.append({"p": p, "day": d + 1 + (1 if rng.random() < 0.3 else 0)})
                        if tr or rng.random() < 0.25:
                            self._log("▣", f"Refill alert → {p.pid} ordered pharmacy delivery")
                    elif tr:
                        self._log("▣", f"Refill alert sent to {p.pid}: 5 days of medicine left")
                if p.supply == 0 and not p.ordered:
                    on_time = 0.9 if p.has("refill") else 0.7
                    if rng.random() < on_time:
                        p.supply = 30
                    else:
                        self.c["run_outs"] += 1
                        p.gap = 1 + rng.randrange(3)
                        if tr:
                            self._log("✗", f"{p.pid} ran out of medicine")
            if rng.random() < p.visits / 365:
                self.c["appts"] += 1
                rem = p.has("appt")
                if rem:
                    self.c["appt_rem"] += 1
                    self._send(p.ch)
                ok = rng.random() < (0.95 if rem else 0.8)
                self.c["appt_attended"] += ok
                if tr:
                    self._log("◷", f"{p.pid} has a doctor appointment today"
                                   f"{' · reminder sent yesterday' if rem else ''}{'' if ok else ' · missed it'}")
            if rng.random() < p.labs / 365:
                self.c["labs"] += 1
                self.c["labs_done"] += rng.random() < (0.93 if p.has("lab") else 0.75)
            if p.has("tele") and rng.random() < 0.004:
                self.c["tele"] += 1
                self.c["revenue"] += sc.tele_fee
                T["rev"] += sc.tele_fee
                if tr:
                    self._log("☏", f"{p.pid} booked a teleconsultation")
            daily_fee = sc.price / 30
            self.c["revenue"] += daily_fee
            T["rev"] += daily_fee
            T["c_pay"] += daily_fee * self.cost.payment_pct + self.cost.payment_fixed / 30
            T["c_sup"] += self.cost.support_month / 30
        self.patients = keep
        alive = set(id(p) for p in keep)
        self.tracked = [p for p in self.tracked if id(p) in alive]
        for p in self.patients:
            if len(self.tracked) >= 8:
                break
            if p not in self.tracked:
                self.tracked.append(p)

    def _close_day(self):
        T = self.today
        for p in self.patients:
            done = [s for s in p.slots if s != "due"]
            if done:
                p.hist.append(sum(s in ("taken", "late", "rescued") for s in done) / len(done))
                p.hist = p.hist[-14:]
        ok = T["taken"] + T["late"] + T["rescued"]
        msg_cost = sum(T["c_msg"])
        cost = msg_cost + T["c_acq"] + T["c_pay"] + T["c_sup"] + T["c_fix"]
        profit = T["rev"] - cost
        self.cash += profit
        if self.cash < self.cash_min:
            self.cash_min, self.cash_min_day = self.cash, self.day - 1
        prev = self.daily[-6:]
        rev7 = (sum(x["rev"] for x in prev) + T["rev"]) / (len(prev) + 1)
        cost7 = (sum(x["cost"] for x in prev) + cost) / (len(prev) + 1)
        if self.profit_day is None and len(self.daily) >= 14 and rev7 > cost7:
            self.profit_day = self.day - 1
        self.daily.append({
            "day": self.day - 1, "active": len(self.patients), "mrr": len(self.patients) * self.sc.price,
            "adherence": ok / T["doses"] if T["doses"] else None,
            "baseline": T["base_sum"] / T["doses"] if T["doses"] else None,
            "new_subs": T["new_subs"], "churn": T["churn"], "leads": T["leads"], "rescued": T["late"] + T["rescued"],
            "rev": T["rev"], "cost": cost, "profit": profit, "cash": self.cash, "rev7": rev7, "cost7": cost7,
            "msg_cost": msg_cost, "c_whatsapp": T["c_msg"][0] + T["c_msg"][1], "c_sms": T["c_msg"][2],
            "c_push": T["c_msg"][3], "c_call": T["c_msg"][4], "c_acq": T["c_acq"], "c_pay": T["c_pay"],
            "c_sup": T["c_sup"], "c_fix": T["c_fix"]})
        self.today = _new_today()

    # ----- summaries for the dashboard
    def daily_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.daily)

    def economics(self, window=30) -> dict:
        w = self.daily[-window:]
        if not w:
            return {}
        s = lambda k: sum(x[k] for x in w)
        rev, msg, acq, pay, sup, fix = s("rev"), s("msg_cost"), s("c_acq"), s("c_pay"), s("c_sup"), s("c_fix")
        cost = msg + acq + pay + sup + fix
        sub_months = s("active") / 30
        contrib = (rev - msg - pay - sup) / sub_months if sub_months else 0.0
        new_subs, churn, leads = s("new_subs"), s("churn"), s("leads")
        cac = acq / new_subs if new_subs else None
        churn_mo = churn / sub_months if sub_months else 0.0
        ltv = contrib / churn_mo if churn_mo > 0 else None
        acq_month = acq * 30 / len(w)
        cover = self.cost.fixed_month + acq_month
        needed = cover / contrib if contrib > 0 else float("inf")
        return {"days": len(w), "rev": rev, "cost": cost, "profit": rev - cost, "msg": msg, "acq": acq, "pay": pay,
                "sup": sup, "fix": fix, "sub_months": sub_months, "contrib": contrib, "cac": cac, "churn_mo": churn_mo,
                "ltv": ltv, "ltv_cac": (ltv / cac) if (ltv and cac) else None,
                "payback": (cac / contrib) if (cac and contrib > 0) else None, "needed": needed,
                "acq_month": acq_month, "cover_month": cover, "conv": new_subs / leads if leads else None,
                "per_sub": {k: (v / sub_months if sub_months else 0) for k, v in
                            {"Revenue": rev, "Messages": -msg, "Payment fees": -pay, "Support": -sup}.items()},
                "breakdown": {"Fixed team & platform": fix, "Lead acquisition": acq,
                              "WhatsApp messages": s("c_whatsapp"), "Automated calls": s("c_call"), "SMS": s("c_sms"),
                              "App notifications": s("c_push"), "Payment fees": pay, "Support": sup}}

    def seven_day_adherence(self):
        w = [x for x in self.daily[-7:] if x["adherence"] is not None]
        if not w:
            return None, None
        return sum(x["adherence"] for x in w) / len(w), sum(x["baseline"] for x in w) / len(w)
