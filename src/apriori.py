"""Apriori frequent itemsets and association rules (no external dependency)."""
from itertools import combinations

import pandas as pd


def frequent_itemsets(basket: pd.DataFrame, min_support: float) -> dict:
    """basket: 0/1 or bool DataFrame (rows = respondents, columns = items). Returns {tuple(items): support}."""
    B = basket.astype(bool)
    items = list(B.columns)

    def support(its):
        return float(B[list(its)].all(axis=1).mean())

    freq = {}
    level = []
    for it in items:
        s = support((it,))
        if s >= min_support:
            freq[(it,)] = s
            level.append((it,))
    k = 2
    while level:
        cands = set()
        for a, b in combinations(level, 2):
            u = tuple(sorted(set(a) | set(b)))
            if len(u) == k and all(tuple(sorted(sub)) in freq for sub in combinations(u, k - 1)):
                cands.add(u)
        level = []
        for c in cands:
            s = support(c)
            if s >= min_support:
                freq[c] = s
                level.append(c)
        k += 1
    return freq


def association_rules(freq: dict, n_rows: int, min_confidence=0.6, min_lift=1.0, single_consequent=True) -> pd.DataFrame:
    rows = []
    for items, sup in freq.items():
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ante in combinations(items, r):
                cons = tuple(i for i in items if i not in ante)
                if single_consequent and len(cons) != 1:
                    continue
                ante_s = tuple(sorted(ante))
                cons_s = tuple(sorted(cons))
                if ante_s not in freq or cons_s not in freq:
                    continue
                conf = sup / freq[ante_s]
                lift = conf / freq[cons_s]
                if conf >= min_confidence and lift >= min_lift:
                    rows.append({"antecedent": ante_s, "consequent": cons_s, "n_antecedent": len(ante_s),
                                 "support": sup, "confidence": conf, "lift": lift, "count": int(round(sup * n_rows))})
    cols = ["antecedent", "consequent", "n_antecedent", "support", "confidence", "lift", "count"]
    return pd.DataFrame(rows, columns=cols).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def lift_matrix(basket: pd.DataFrame) -> pd.DataFrame:
    B = basket.astype(bool)
    sup = B.mean()
    cols = list(B.columns)
    m = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            m.loc[a, b] = float("nan") if a == b else (B[a] & B[b]).mean() / (sup[a] * sup[b])
    return m
