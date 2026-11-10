"""EDA: six questions, tables (CSV) and figures (PNG). Delivery outcomes are used descriptively only."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .modeling import temporal_split
from .numpy_ops import days_between, percentile_bounds

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "figure.dpi": 100})
BLUE, RED, GRAY = "#2a6f97", "#c1121f", "#9aa5b1"


def _wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def late_table(df, col, min_n=C.MIN_GROUP_N):
    g = df.groupby(col, observed=True)["is_late"].agg(n="size", n_late="sum").reset_index()
    g["late_rate"] = g.n_late / g.n
    lo, hi = _wilson(g.n_late, g.n)
    g["ci95_low"], g["ci95_high"] = lo, hi
    g["n_ge_min"] = g.n >= min_n
    return g


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=200)
    plt.close(fig)


def _bar_with_n(ax, t, label_col, overall, horizontal=True, color=BLUE):
    t = t.copy()
    if horizontal:
        y = np.arange(len(t))
        ax.barh(y, t.late_rate * 100, color=color, xerr=[(t.late_rate - t.ci95_low) * 100, (t.ci95_high - t.late_rate) * 100],
                error_kw={"ecolor": "#555", "lw": 1})
        ax.set_yticks(y, t[label_col].astype(str))
        for yi, (r, n) in enumerate(zip(t.late_rate, t.n)):
            ax.text(t.ci95_high.iloc[yi] * 100 + 0.2, yi, f"{r*100:.1f}%  (n={n:,})", va="center", fontsize=8)
        ax.axvline(overall * 100, color=RED, ls="--", lw=1)
        ax.set_xlabel("Late-delivery rate (%)  | bars: 95% Wilson CI | dashed = overall rate")
        ax.invert_yaxis()
    else:
        x = np.arange(len(t))
        ax.bar(x, t.late_rate * 100, color=color, yerr=[(t.late_rate - t.ci95_low) * 100, (t.ci95_high - t.late_rate) * 100],
               error_kw={"ecolor": "#555", "lw": 1})
        ax.set_xticks(x, t[label_col].astype(str), rotation=0)
        for xi, (r, n) in enumerate(zip(t.late_rate, t.n)):
            ax.text(xi, t.ci95_high.iloc[xi] * 100 + 0.15, f"{r*100:.1f}%\nn={n:,}", ha="center", fontsize=8)
        ax.axhline(overall * 100, color=RED, ls="--", lw=1)
        ax.set_ylim(0, t.ci95_high.max() * 100 * 1.3)
        ax.set_ylabel("Late-delivery rate (%)")


def run_eda(full: pd.DataFrame, e: pd.DataFrame, funnel: pd.DataFrame) -> dict:
    """full = all 99,441 orders with features; e = eligible orders (model table)."""
    e = e.copy()
    overall = e.is_late.mean()
    N = len(e)
    pop = f"Population: eligible delivered orders, n={N:,}"
    keys = {"n_all_orders": int(len(full)), "n_eligible": int(N), "late_rate_overall": float(overall), "n_late": int(e.is_late.sum())}

    # =============== Q1: time ===============
    e["purchase_month"] = e.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
    full["purchase_month"] = full.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
    cnt_all = full.groupby("purchase_month").size().rename("orders_all")
    lt = late_table(e, "purchase_month").set_index("purchase_month")
    q1 = pd.concat([cnt_all, lt], axis=1).reset_index()
    q1.to_csv(C.TABLE_DIR / "q1_monthly_orders_late_rate.csv", index=False)
    ok = q1[q1.n >= C.MIN_GROUP_N]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(q1.purchase_month, q1.orders_all, width=25, color=GRAY, label="All orders (any status)")
    ax.set_ylabel("Orders per purchase month"); ax.set_xlabel("Purchase month")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(ok.purchase_month, ok.late_rate * 100, color=RED, marker="o", label=f"Late rate (eligible, n>={C.MIN_GROUP_N} months)")
    ax2.set_ylabel("Late rate (%)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title(f"Monthly order volume and late-delivery rate\nBars: all {len(full):,} orders; line: eligible orders ({N:,}); months with n<{C.MIN_GROUP_N} not plotted")
    _save(fig, "fig01_monthly_orders_and_late_rate.png")
    peak = ok.loc[ok.late_rate.idxmax()]
    keys["peak_late_month"] = str(peak.purchase_month.date())[:7]; keys["peak_late_rate"] = float(peak.late_rate); keys["peak_late_n"] = int(peak.n)
    keys["min_late_month"] = str(ok.loc[ok.late_rate.idxmin()].purchase_month.date())[:7]; keys["min_late_rate"] = float(ok.late_rate.min())

    # =============== target distribution (class imbalance) ===============
    tr, va, te = temporal_split(e)
    dist = pd.DataFrame([{"population": n, "n": len(d), "late": int(d.is_late.sum()), "on_time": int(len(d) - d.is_late.sum()),
                          "late_rate": d.is_late.mean()} for n, d in (("all eligible", e), ("train", tr), ("validation", va), ("test", te))])
    dist.to_csv(C.TABLE_DIR / "target_distribution.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["On time (0)", "Late (1)"], [N - e.is_late.sum(), e.is_late.sum()], color=[BLUE, RED])
    for i, v in enumerate([N - e.is_late.sum(), e.is_late.sum()]):
        axes[0].text(i, v, f"{v:,}\n({v/N*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Orders"); axes[0].set_title(f"Target distribution\n{pop}")
    axes[0].set_ylim(0, N * 1.15)
    axes[1].bar(dist.population[1:], dist.late_rate[1:] * 100, color=[BLUE, BLUE, RED])
    for i, (r, n) in enumerate(zip(dist.late_rate[1:], dist.n[1:])):
        axes[1].text(i, r * 100 + 0.1, f"{r*100:.1f}%\nn={n:,}", ha="center", fontsize=9)
    axes[1].set_ylabel("Late rate (%)"); axes[1].set_title("Late rate per temporal split"); axes[1].set_ylim(0, dist.late_rate[1:].max() * 100 * 1.4)
    _save(fig, "fig02_target_distribution.png")

    # =============== Q2: routes / multi-seller ===============
    st = late_table(e, "customer_state").sort_values("late_rate", ascending=False)
    st.to_csv(C.TABLE_DIR / "q2_late_by_customer_state.csv", index=False)
    stp = st[st.n_ge_min]
    fig, ax = plt.subplots(figsize=(9, 7))
    _bar_with_n(ax, stp, "customer_state", overall)
    ax.set_title(f"Late rate by destination (customer) state - states with n>={C.MIN_GROUP_N}\n{pop}; {len(st)-len(stp)} small state(s) omitted")
    ax.set_xlim(0, (stp.ci95_high.max() * 100) * 1.35)
    _save(fig, "fig03_late_rate_by_customer_state.png")
    keys["worst_state"] = stp.iloc[0].customer_state; keys["worst_state_rate"] = float(stp.iloc[0].late_rate); keys["worst_state_n"] = int(stp.iloc[0].n)
    keys["best_state"] = stp.iloc[-1].customer_state; keys["best_state_rate"] = float(stp.iloc[-1].late_rate); keys["best_state_n"] = int(stp.iloc[-1].n)
    keys["n_states_small"] = int(len(st) - len(stp))

    # route heatmap
    top_s = e.seller_state.value_counts().head(6).index
    top_c = e.customer_state.value_counts().head(10).index
    rt = e[e.seller_state.isin(top_s) & e.customer_state.isin(top_c)]
    route = late_table(rt, ["seller_state", "customer_state"])
    route.to_csv(C.TABLE_DIR / "q2_route_late_rate.csv", index=False)
    rate = route.pivot(index="seller_state", columns="customer_state", values="late_rate").reindex(index=top_s, columns=top_c)
    nn = route.pivot(index="seller_state", columns="customer_state", values="n").reindex(index=top_s, columns=top_c)
    show = rate.where(nn >= C.MIN_GROUP_N) * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(show.values, cmap="Reds", aspect="auto", vmin=0)
    ax.set_xticks(range(len(top_c)), top_c); ax.set_yticks(range(len(top_s)), top_s)
    for i in range(show.shape[0]):
        for j in range(show.shape[1]):
            n = nn.values[i, j]
            if np.isnan(n) or n < C.MIN_GROUP_N:
                ax.text(j, i, f"n={0 if np.isnan(n) else int(n)}\n(<{C.MIN_GROUP_N})", ha="center", va="center", fontsize=6.5, color="#888")
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True, color="#eee", zorder=0))
            else:
                ax.text(j, i, f"{show.values[i, j]:.1f}%\nn={int(n):,}", ha="center", va="center", fontsize=7.5)
    ax.set_xlabel("Customer state (destination)"); ax.set_ylabel("Primary seller state (origin)")
    ax.set_title(f"Late rate (%) by route - top 6 seller states x top 10 customer states\n{pop}; cells with n<{C.MIN_GROUP_N} greyed out")
    fig.colorbar(im, ax=ax, label="Late rate (%)")
    _save(fig, "fig04_route_heatmap.png")

    ms = late_table(e.assign(group=np.where(e.multi_seller == 1, "Multi-seller", "Single seller")), "group")
    ss = late_table(e.assign(group=np.where(e.same_state == 1, "Same state", "Different state")), "group")
    ms.to_csv(C.TABLE_DIR / "q2_multi_seller.csv", index=False); ss.to_csv(C.TABLE_DIR / "q2_same_state.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _bar_with_n(axes[0], ms, "group", overall, horizontal=False); axes[0].set_title("Single vs multi-seller orders")
    _bar_with_n(axes[1], ss, "group", overall, horizontal=False, color="#6a994e"); axes[1].set_title("Seller and customer in same state?")
    fig.suptitle(f"{pop}; error bars 95% Wilson CI; dashed = overall", fontsize=9)
    _save(fig, "fig05_multiseller_and_same_state.png")
    for tab, nm in ((ms, "multi"), (ss, "same")):
        for r in tab.itertuples():
            keys[f"{nm}_{r.group.replace(' ', '_').lower()}_rate"] = float(r.late_rate); keys[f"{nm}_{r.group.replace(' ', '_').lower()}_n"] = int(r.n)

    # =============== Q3: categories and size ===============
    cat = late_table(e, "main_category").sort_values("late_rate", ascending=False)
    cat.to_csv(C.TABLE_DIR / "q3_late_by_category.csv", index=False)
    catp = cat[cat.n_ge_min]
    sel = pd.concat([catp.head(8), catp.tail(8)]).drop_duplicates("main_category")
    fig, ax = plt.subplots(figsize=(9.5, 7))
    _bar_with_n(ax, sel, "main_category", overall)
    ax.set_title(f"Late rate by dominant product category - 8 highest and 8 lowest of {len(catp)} categories with n>={C.MIN_GROUP_N}\n{pop}; {len(cat)-len(catp)} small categories excluded")
    ax.set_xlim(0, sel.ci95_high.max() * 100 * 1.4)
    _save(fig, "fig06_late_rate_by_category.png")
    keys["cat_worst"] = catp.iloc[0].main_category; keys["cat_worst_rate"] = float(catp.iloc[0].late_rate); keys["cat_worst_n"] = int(catp.iloc[0].n)
    keys["cat_best"] = catp.iloc[-1].main_category; keys["cat_best_rate"] = float(catp.iloc[-1].late_rate); keys["cat_best_n"] = int(catp.iloc[-1].n)
    keys["n_categories_total"] = int(len(cat)); keys["n_categories_ge_min"] = int(len(catp))

    e["size_bin"] = pd.cut(e.n_items, [0, 1, 2, 3, 100], labels=["1 item", "2 items", "3 items", "4+ items"])
    e["price_bin"] = pd.qcut(e.total_price, 5, duplicates="drop")
    e["price_bin"] = e["price_bin"].astype(str)
    sz = late_table(e, "size_bin"); pr = late_table(e, "price_bin")
    pr["_o"] = pr.price_bin.str.extract(r"\((.*?),")[0].astype(float); pr = pr.sort_values("_o").drop(columns="_o")
    sz.to_csv(C.TABLE_DIR / "q3_late_by_order_size.csv", index=False); pr.to_csv(C.TABLE_DIR / "q3_late_by_order_value.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _bar_with_n(axes[0], sz, "size_bin", overall, horizontal=False); axes[0].set_title("By number of items in the order"); axes[0].set_xlabel("Items per order")
    _bar_with_n(axes[1], pr, "price_bin", overall, horizontal=False, color="#6a994e"); axes[1].set_title("By order value quintile (BRL, total item price)")
    axes[1].tick_params(axis="x", labelsize=7); axes[1].set_xlabel("Total price range (BRL)")
    fig.suptitle(pop, fontsize=9)
    _save(fig, "fig07_late_rate_by_order_size_and_value.png")
    keys["size_rates"] = {str(r.size_bin): [float(r.late_rate), int(r.n)] for r in sz.itertuples()}

    # =============== Q4: planned time vs actual ===============
    e["actual_days"] = days_between(e.order_delivered_customer_date, e.order_approved_at)  # outcome: descriptive only
    lo, hi = percentile_bounds(e.planned_days, 0.5, 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(lo, hi, 45)
    for v, c, lab in ((0, BLUE, "On time"), (1, RED, "Late")):
        axes[0].hist(e.loc[e.is_late == v, "planned_days"].clip(lo, hi), bins=bins, alpha=.6, color=c, density=True, label=f"{lab} (n={int((e.is_late==v).sum()):,})")
    axes[0].set_xlabel("Planned delivery window: estimated date - approval (days)"); axes[0].set_ylabel("Density"); axes[0].legend()
    axes[0].set_title("Distribution of planned days by outcome (clipped to 0.5-99.5 pct)")
    data = [e.loc[e.is_late == 0, "planned_days"].clip(lo, hi), e.loc[e.is_late == 1, "planned_days"].clip(lo, hi)]
    bp = axes[1].boxplot(data, tick_labels=["On time", "Late"], patch_artist=True, showfliers=False)
    for p_, c in zip(bp["boxes"], (BLUE, RED)): p_.set_facecolor(c); p_.set_alpha(.6)
    axes[1].set_ylabel("Planned days"); axes[1].set_title("Planned days: box plot by outcome")
    fig.suptitle(pop, fontsize=9)
    _save(fig, "fig08_planned_days_by_outcome.png")
    ptab = e.groupby("is_late").planned_days.describe().reset_index(); ptab.to_csv(C.TABLE_DIR / "q4_planned_days_by_outcome.csv", index=False)
    keys["planned_median_ontime"] = float(e.loc[e.is_late == 0, "planned_days"].median()); keys["planned_median_late"] = float(e.loc[e.is_late == 1, "planned_days"].median())

    e["planned_bin"] = pd.qcut(e.planned_days, 8, duplicates="drop").astype(str)
    pb = late_table(e, "planned_bin"); pb["_o"] = pb.planned_bin.str.extract(r"\((.*?),")[0].astype(float); pb = pb.sort_values("_o").drop(columns="_o")
    pb.to_csv(C.TABLE_DIR / "q4_late_by_planned_days_octile.csv", index=False)
    samp = e.sample(min(15000, N), random_state=C.SEED)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(samp.planned_days[samp.is_late == 0], samp.actual_days[samp.is_late == 0], s=3, alpha=.25, color=BLUE, label="On time")
    axes[0].scatter(samp.planned_days[samp.is_late == 1], samp.actual_days[samp.is_late == 1], s=4, alpha=.5, color=RED, label="Late")
    mx = float(np.nanpercentile(e.planned_days, 99.5))
    axes[0].plot([0, mx], [0, mx], color="k", ls="--", lw=1, label="actual = planned")
    axes[0].set_xlim(0, mx); axes[0].set_ylim(0, 80)
    axes[0].set_xlabel("Planned days (approval to estimated date)"); axes[0].set_ylabel("Actual days (approval to delivery)")
    axes[0].set_title(f"Planned vs actual duration (random sample of {len(samp):,} orders)"); axes[0].legend(markerscale=4)
    _bar_with_n(axes[1], pb, "planned_bin", overall, horizontal=False); axes[1].tick_params(axis="x", labelsize=6, rotation=30)
    axes[1].set_xlabel("Planned days (octile bins)"); axes[1].set_title("Late rate by planned-days octile")
    fig.suptitle(pop + "  (actual days = outcome, descriptive use only)", fontsize=9)
    _save(fig, "fig09_planned_vs_actual.png")
    keys["corr_planned_actual"] = float(e[["planned_days", "actual_days"]].corr(method="spearman").iloc[0, 1])
    keys["planned_octile_rates"] = [[str(r.planned_bin), float(r.late_rate), int(r.n)] for r in pb.itertuples()]

    # =============== relationships: correlation heatmap ===============
    num = F.NUMERIC + ["is_late"]
    corr = e[num].astype(float).corr(method="spearman")
    corr.to_csv(C.TABLE_DIR / "correlation_spearman.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)), num, rotation=90, fontsize=7); ax.set_yticks(range(len(num)), num, fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title(f"Spearman correlation: numeric features and is_late\n{pop}. Correlation is not causation")
    _save(fig, "fig10_correlation_heatmap.png")
    ct = corr["is_late"].drop("is_late").sort_values(key=np.abs, ascending=False)
    ct.rename("spearman_with_is_late").to_csv(C.TABLE_DIR / "correlation_with_target.csv")
    keys["top_corr"] = {k: float(v) for k, v in ct.head(5).items()}

    # =============== Q5: missingness / exclusions ===============
    status = full.order_status.value_counts().rename_axis("order_status").reset_index(name="n")
    status["eligible"] = status.order_status.map(full.groupby("order_status").eligible.sum()).astype(int)
    status.to_csv(C.TABLE_DIR / "q5_status_vs_eligible.csv", index=False)
    miss_cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "n_items", "total_payment", "total_weight_g"]
    miss = pd.DataFrame({"column": miss_cols, "missing_all_orders": [int(full[c].isna().sum()) for c in miss_cols],
                         "missing_eligible": [int(e[c].isna().sum()) for c in miss_cols]})
    miss.to_csv(C.TABLE_DIR / "q5_missing_before_after.csv", index=False)
    comp_cols = {"total_price": "median", "planned_days": "median", "n_items": "mean", "multi_seller": "mean", "same_state": "mean"}
    inel = full[~full.eligible]
    rep = pd.DataFrame({"feature": list(comp_cols), "stat": list(comp_cols.values()),
                        "eligible": [getattr(e[c], s)() for c, s in comp_cols.items()],
                        "excluded": [getattr(inel[c], s)() for c, s in comp_cols.items()]})
    rep.to_csv(C.TABLE_DIR / "q5_eligible_vs_excluded_profile.csv", index=False)
    top_states = full.customer_state.value_counts().head(8).index
    sh = pd.DataFrame({"all_orders": full.customer_state.value_counts(normalize=True), "eligible": e.customer_state.value_counts(normalize=True)}).loc[top_states] * 100
    sh.to_csv(C.TABLE_DIR / "q5_state_share_all_vs_eligible.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    f2 = funnel.copy(); f2["short"] = ["All", "delivered", "dates ok", "ts valid", "deliv>=purch", ">=1 item"]
    axes[0].bar(f2.short, f2.orders_remaining, color=BLUE)
    for i, v in enumerate(f2.orders_remaining): axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Orders remaining"); axes[0].set_title("Eligibility funnel (all 99,441 orders -> eligible)"); axes[0].tick_params(axis="x", labelsize=7, rotation=20)
    stt = status.sort_values("n", ascending=False)
    axes[1].barh(stt.order_status, stt.n, color=GRAY); axes[1].barh(stt.order_status, stt.eligible, color=BLUE)
    axes[1].set_xscale("log"); axes[1].invert_yaxis(); axes[1].set_xlabel("Orders (log scale)"); axes[1].set_title("Status: all (grey) vs eligible (blue)")
    x = np.arange(len(top_states)); w = .4
    axes[2].bar(x - w/2, sh.all_orders, w, color=GRAY, label="All orders"); axes[2].bar(x + w/2, sh.eligible, w, color=BLUE, label="Eligible")
    axes[2].set_xticks(x, top_states); axes[2].set_ylabel("Share of orders (%)"); axes[2].set_title("Top-8 customer states: share before vs after"); axes[2].legend()
    _save(fig, "fig11_exclusions_impact.png")
    keys["n_excluded"] = int(len(inel)); keys["funnel"] = funnel[["step", "orders_remaining"]].to_dict("records")

    # =============== Q6 key numbers: pooled patterns ===============
    keys["train_late"], keys["val_late"], keys["test_late"] = float(tr.is_late.mean()), float(va.is_late.mean()), float(te.is_late.mean())
    (C.TABLE_DIR / "eda_key_numbers.json").write_text(json.dumps(keys, indent=2, default=str), encoding="utf-8")
    return keys
