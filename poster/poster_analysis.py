"""
Statistical Poster Analysis — BugsRepo (Mozilla Bugzilla) metadata
==================================================================
Activity 1: Statistical Poster Presentation.

Run:  python poster_analysis.py

Produces, in ./figures/ :
    fig1_comment_count_distribution.png   histogram + boxplot, quartile markers
    fig2_resolution_time_by_severity.png  grouped box plot
    fig3_correlation_heatmap.png          Spearman correlation matrix
    fig4_scatter_comments_vs_time.png     scatter with regression, log-log
    fig5_summary_length_distribution.png  histogram + KDE, mean/median/percentiles
    fig6_severity_priority_composition.png stacked composition + counts

and  stats_summary.txt  — every number you need to read off the charts.

NOTE: this script computes and plots. The written observations on the poster
are yours to make from these numbers.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
RAW = Path(r"C:\Users\abhin\bugtriage-data\raw\bugsrepo\Bug_meta_data.csv")
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
DPI = 300                      # print quality for A3
SEED = 42
rng = np.random.default_rng(SEED)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"S1": "#b2182b", "S2": "#ef8a62", "S3": "#67a9cf", "S4": "#2166ac"}

USECOLS = ["id", "severity", "priority", "product", "component", "type", "status",
           "resolution", "creation_time", "cf_last_resolved", "comment_count",
           "votes", "summary", "cc", "blocks", "depends_on", "duplicates",
           "keywords", "cf_crash_signature", "dupe_of"]

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# ----------------------------------------------------------------------------
# Load & clean
# ----------------------------------------------------------------------------
log("=" * 70)
log("LOADING")
log("=" * 70)
df = pd.read_csv(RAW, usecols=lambda c: c in USECOLS, low_memory=False)
log(f"raw rows                : {len(df):,}")

df = df.drop_duplicates(subset=["id"], keep="first").copy()
log(f"after de-duplicating id : {len(df):,}")

# --- derived: severity (keep only the modern S1-S4 scale) -------------------
df["severity"] = df["severity"].astype("string").str.strip()
sev_modern = df["severity"].isin(["S1", "S2", "S3", "S4"])
log(f"rows with S1-S4 severity: {int(sev_modern.sum()):,} "
    f"({sev_modern.mean() * 100:.1f}%)")

# --- derived: resolution time ----------------------------------------------
ct = pd.to_datetime(df["creation_time"], errors="coerce", format="mixed", utc=True)
lr = pd.to_datetime(df["cf_last_resolved"], errors="coerce", format="mixed", utc=True)
df["creation_time"] = ct
df["resolution_days"] = (lr - ct).dt.total_seconds() / 86400.0
df.loc[df["resolution_days"] < 0, "resolution_days"] = np.nan
log(f"resolution_days available: {int(df['resolution_days'].notna().sum()):,}")

# --- derived: text length ---------------------------------------------------
df["summary"] = df["summary"].astype("string").fillna("")
df["summary_len_chars"] = df["summary"].str.len()
df["summary_len_words"] = df["summary"].str.split().str.len().fillna(0)


# --- derived: list-column lengths ------------------------------------------
def list_len(s):
    """Number of elements in a stringified python list.

    Works for numeric lists ('[1, 2, 3]') and string lists
    ("['a@b.com', 'c@d.com']") alike: strip the brackets, and if anything
    is left, the element count is (number of commas + 1).
    """
    inner = (s.astype("string").fillna("[]").str.strip()
             .str.removeprefix("[").str.removesuffix("]").str.strip())
    return np.where(inner.str.len() == 0, 0, inner.str.count(",") + 1).astype(int)


for col in ["blocks", "depends_on", "duplicates", "cc"]:
    if col in df.columns:
        df[f"n_{col}"] = list_len(df[col])

df["n_keywords"] = (df["keywords"].astype("string").fillna("")
                    .apply(lambda s: 0 if not s.strip() else len(re.split(r"[,\s]+", s.strip()))))
df["has_crash_sig"] = df["cf_crash_signature"].notna().astype(int)

# ----------------------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------------------
log()
log("=" * 70)
log("DESCRIPTIVE STATISTICS  (central tendency, spread, shape)")
log("=" * 70)

NUMERIC = ["comment_count", "resolution_days", "summary_len_chars",
           "summary_len_words", "n_blocks", "n_depends_on", "n_cc",
           "n_keywords", "votes"]

desc = df[NUMERIC].describe(percentiles=[.05, .25, .5, .75, .90, .95, .99]).T
desc["IQR"] = desc["75%"] - desc["25%"]
desc["skew"] = df[NUMERIC].skew()
desc["kurtosis"] = df[NUMERIC].kurtosis()
log(desc.round(2).to_string())

log()
log("Outlier counts by the 1.5 x IQR rule (Tukey):")
for c in NUMERIC:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    n_out = int((df[c] > hi).sum())
    log(f"  {c:20s} Q1={q1:9.2f}  Q3={q3:9.2f}  IQR={iqr:9.2f}  "
        f"upper fence={hi:9.2f}  outliers={n_out:,} ({n_out / df[c].notna().sum() * 100:.2f}%)")

# ----------------------------------------------------------------------------
# FIGURE 1 — distribution of comment_count (histogram + box plot)
# ----------------------------------------------------------------------------
d = df["comment_count"].dropna()
fig, (ax_box, ax_hist) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={"height_ratios": (.25, .75)})

sns.boxplot(x=d, ax=ax_box, color="#67a9cf", fliersize=1.5, width=.5)
ax_box.set(xlabel="", yticks=[])
ax_box.set_title("Distribution of Comments per Bug Report", pad=14, weight="bold")

bins = np.arange(0, 41) - .5
ax_hist.hist(d.clip(upper=40), bins=bins, color="#67a9cf",
             edgecolor="white", linewidth=.6)
for val, col, ls, lab in [(d.mean(), "#b2182b", "-", f"Mean = {d.mean():.2f}"),
                          (d.median(), "#1a9850", "--", f"Median = {d.median():.0f}"),
                          (d.quantile(.25), "#666666", ":", f"Q1 = {d.quantile(.25):.0f}"),
                          (d.quantile(.75), "#666666", "-.", f"Q3 = {d.quantile(.75):.0f}")]:
    ax_hist.axvline(val, color=col, linestyle=ls, linewidth=2, label=lab)
ax_hist.set_xlabel("Number of comments  (clipped at 40 for display)")
ax_hist.set_ylabel("Number of bugs")
ax_hist.legend(frameon=True, fontsize=11)
ax_hist.set_xlim(-.5, 40)
fig.tight_layout()
fig.savefig(OUT / "fig1_comment_count_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

log()
log("FIG 1  comment_count:")
log(f"  n={len(d):,}  mean={d.mean():.2f}  median={d.median():.0f}  "
    f"mode={d.mode().iloc[0]}  std={d.std():.2f}")
log(f"  min={d.min()}  Q1={d.quantile(.25):.0f}  Q3={d.quantile(.75):.0f}  "
    f"P90={d.quantile(.90):.0f}  P99={d.quantile(.99):.0f}  max={d.max()}")
log(f"  skewness={d.skew():.2f}   (>0 => right-tailed)")

# ----------------------------------------------------------------------------
# FIGURE 2 — resolution time by severity (box plot)
# ----------------------------------------------------------------------------
# Exclude sub-15-minute "resolutions": these are automated closures (spam,
# instant duplicates, bot actions), not genuine triage-and-fix cycles. Keeping
# them stretches the log axis over 7 decades and hides the real group differences.
MIN_DAYS = 0.01                                   # ~15 minutes
_res = df[sev_modern & df["resolution_days"].notna() & (df["resolution_days"] > 0)]
box = _res[_res["resolution_days"] >= MIN_DAYS].copy()
order = ["S1", "S2", "S3", "S4"]
log()
log(f"FIG 2  excluded {len(_res) - len(box):,} of {len(_res):,} resolved bugs "
    f"({(len(_res) - len(box)) / len(_res) * 100:.2f}%) with resolution time "
    f"< {MIN_DAYS} days (~15 min) as automated closures.")

fig, ax = plt.subplots(figsize=(9, 6.5))
sns.boxplot(data=box, x="severity", y="resolution_days", order=order,
            hue="severity", palette=PALETTE, legend=False, ax=ax,
            fliersize=1, linewidth=1.4, showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white",
                       "markeredgecolor": "black", "markersize": 7})
ax.set_yscale("log")
ax.set_xlabel("Severity  (S1 = most severe  →  S4 = least severe)")
ax.set_ylabel("Time to resolution (days, log scale)")
ax.set_title("Bug Resolution Time by Severity Level", pad=14, weight="bold")
for i, s in enumerate(order):
    sub = box.loc[box["severity"] == s, "resolution_days"]
    ax.text(i, box["resolution_days"].max() * 1.4, f"n={len(sub):,}",
            ha="center", fontsize=11, color="#333333")
fig.tight_layout()
fig.savefig(OUT / "fig2_resolution_time_by_severity.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

log()
log("FIG 2  resolution_days by severity (five-number summary):")
grp = box.groupby("severity", observed=True)["resolution_days"]
summ = grp.describe(percentiles=[.25, .5, .75])[["count", "min", "25%", "50%", "75%", "max", "mean"]]
summ["IQR"] = summ["75%"] - summ["25%"]
log(summ.round(2).to_string())

kw = stats.kruskal(*[g.values for _, g in grp])
log(f"  Kruskal-Wallis H={kw.statistic:.1f}, p={kw.pvalue:.3e}  "
    f"(tests whether medians differ across the four groups)")

# ----------------------------------------------------------------------------
# FIGURE 3 — correlation heatmap
# ----------------------------------------------------------------------------
CORR_COLS = ["comment_count", "resolution_days", "summary_len_words",
             "n_blocks", "n_depends_on", "n_cc", "n_keywords", "votes"]
LABELS = ["Comments", "Resolution\ndays", "Summary\nwords", "Blocks",
          "Depends on", "CC list\nsize", "Keywords", "Votes"]

corr = df[CORR_COLS].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(9.5, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=.8,
            cbar_kws={"shrink": .8, "label": "Spearman ρ"},
            xticklabels=LABELS, yticklabels=LABELS, ax=ax,
            annot_kws={"size": 12})
ax.set_title("Correlation Between Bug Report Attributes\n(Spearman rank correlation)",
             pad=16, weight="bold")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(OUT / "fig3_correlation_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

log()
log("FIG 3  Spearman correlation matrix:")
log(corr.round(3).to_string())
log()
log("  Strongest pairs (|rho|, excluding self-correlation):")
pairs = (corr.where(mask.T & ~np.eye(len(corr), dtype=bool))
         .stack().abs().sort_values(ascending=False).head(6))
for (a, b), v in pairs.items():
    log(f"    {a:18s} ~ {b:18s} rho = {corr.loc[a, b]:+.3f}")

# ----------------------------------------------------------------------------
# FIGURE 4 — scatter: comments vs resolution time
# ----------------------------------------------------------------------------
sc = box[(box["comment_count"] > 0)].copy()
if len(sc) > 12000:                       # keep the figure readable & the file small
    sc = sc.sample(12000, random_state=SEED)

fig, ax = plt.subplots(figsize=(9.5, 7))
for s in order:
    sub = sc[sc["severity"] == s]
    ax.scatter(sub["comment_count"], sub["resolution_days"], s=9, alpha=.35,
               color=PALETTE[s], label=f"{s} (n={len(sub):,})", edgecolors="none")
ax.set_xscale("log"); ax.set_yscale("log")

x = np.log10(sc["comment_count"]); y = np.log10(sc["resolution_days"])
ok = np.isfinite(x) & np.isfinite(y)
slope, icept, r, p, se = stats.linregress(x[ok], y[ok])
xs = np.linspace(x[ok].min(), x[ok].max(), 100)
ax.plot(10 ** xs, 10 ** (icept + slope * xs), color="black", linewidth=2.2,
        label=f"Least-squares fit (log-log)")

rho, rho_p = stats.spearmanr(sc["comment_count"], sc["resolution_days"])
ax.set_xlabel("Number of comments (log scale)")
ax.set_ylabel("Time to resolution in days (log scale)")
ax.set_title("Discussion Volume vs. Time to Resolution", pad=14, weight="bold")
ax.legend(frameon=True, fontsize=10, loc="lower right")
ax.text(.03, .97, f"Spearman ρ = {rho:.3f}\np < 1e-300" if rho_p < 1e-300
        else f"Spearman ρ = {rho:.3f}\np = {rho_p:.2e}",
        transform=ax.transAxes, va="top", fontsize=13,
        bbox=dict(boxstyle="round,pad=.5", facecolor="white", edgecolor="#999999"))
fig.tight_layout()
fig.savefig(OUT / "fig4_scatter_comments_vs_time.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

log()
log("FIG 4  comments vs resolution time:")
log(f"  n plotted={len(sc):,}  Spearman rho={rho:.4f}  p={rho_p:.3e}")
log(f"  log-log least squares: slope={slope:.4f}  intercept={icept:.4f}  "
    f"r={r:.4f}  r^2={r ** 2:.4f}")

# ----------------------------------------------------------------------------
# FIGURE 5 — summary length distribution
# ----------------------------------------------------------------------------
sl = df["summary_len_words"]
sl = sl[(sl > 0) & (sl <= 40)]

fig, ax = plt.subplots(figsize=(9.5, 6.5))
ax.hist(sl, bins=np.arange(0, 41) - .5, density=True, color="#c7e9c0",
        edgecolor="#4a7c59", linewidth=.7, label="Observed distribution")
sl.plot.kde(ax=ax, color="#1a9850", linewidth=2.5, label="Kernel density estimate")

qs = df["summary_len_words"].quantile([.25, .5, .75, .90])
ax.axvline(df["summary_len_words"].mean(), color="#b2182b", linewidth=2.2,
           label=f"Mean = {df['summary_len_words'].mean():.2f} words")
ax.axvline(qs[.5], color="#333333", linestyle="--", linewidth=2.2,
           label=f"Median = {qs[.5]:.0f} words")
for q, ls in [(.25, ":"), (.75, ":"), (.90, "-.")]:
    ax.axvline(qs[q], color="#777777", linestyle=ls, linewidth=1.6,
               label=f"P{int(q * 100)} = {qs[q]:.0f} words")
ax.set_xlim(0, 40)
ax.set_xlabel("Length of bug report title (words)")
ax.set_ylabel("Probability density")
ax.set_title("Distribution of Bug Report Title Length", pad=14, weight="bold")
ax.legend(frameon=True, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig5_summary_length_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

full_sl = df["summary_len_words"]
log()
log("FIG 5  summary length (words):")
log(f"  n={full_sl.notna().sum():,}  mean={full_sl.mean():.2f}  "
    f"median={full_sl.median():.0f}  std={full_sl.std():.2f}  "
    f"skew={full_sl.skew():.2f}  kurtosis={full_sl.kurtosis():.2f}")
log(f"  P5={full_sl.quantile(.05):.0f}  Q1={full_sl.quantile(.25):.0f}  "
    f"Q3={full_sl.quantile(.75):.0f}  P90={full_sl.quantile(.90):.0f}  "
    f"P99={full_sl.quantile(.99):.0f}  max={full_sl.max():.0f}")
sh = stats.shapiro(full_sl.dropna().sample(4000, random_state=SEED))
log(f"  Shapiro-Wilk normality test on n=4000 sample: W={sh.statistic:.4f}, "
    f"p={sh.pvalue:.3e}")

# ----------------------------------------------------------------------------
# FIGURE 6 — severity composition & label completeness
# ----------------------------------------------------------------------------
sev_all = df["severity"].fillna("(missing)").replace({"--": "-- (unset)"})
counts = sev_all.value_counts()
keep = ["-- (unset)", "S4", "S3", "(missing)", "S2", "S1"]
counts = counts[[k for k in keep if k in counts.index]]
pct = counts / counts.sum() * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

colors = ["#bdbdbd", "#2166ac", "#67a9cf", "#e0e0e0", "#ef8a62", "#b2182b"]
bars = ax1.bar(range(len(counts)), counts.values,
               color=colors[:len(counts)], edgecolor="white", linewidth=1.2)
ax1.set_xticks(range(len(counts)))
ax1.set_xticklabels(counts.index, rotation=20, ha="right")
ax1.set_ylabel("Number of bug reports")
ax1.set_title("Severity Label Availability", pad=12, weight="bold")
for b, v, p in zip(bars, counts.values, pct.values):
    ax1.text(b.get_x() + b.get_width() / 2, v + counts.max() * .015,
             f"{v:,}\n({p:.1f}%)", ha="center", fontsize=10.5)
ax1.set_ylim(0, counts.max() * 1.18)

# labelled subset only
lab = df.loc[sev_modern, "severity"].value_counts().reindex(order)
lab_pct = lab / lab.sum() * 100
bars2 = ax2.bar(range(4), lab.values, color=[PALETTE[s] for s in order],
                edgecolor="white", linewidth=1.2)
ax2.set_xticks(range(4)); ax2.set_xticklabels(order)
ax2.set_ylabel("Number of bug reports")
ax2.set_xlabel("Severity")
ax2.set_title("Class Balance Among Labelled Bugs Only", pad=12, weight="bold")
for b, v, p in zip(bars2, lab.values, lab_pct.values):
    ax2.text(b.get_x() + b.get_width() / 2, v + lab.max() * .015,
             f"{v:,}\n({p:.1f}%)", ha="center", fontsize=11)
ax2.set_ylim(0, lab.max() * 1.18)
fig.suptitle("Severity Distribution and the Missing-Label Problem",
             fontsize=17, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig6_severity_priority_composition.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

log()
log("FIG 6  severity composition (all rows):")
log(pd.DataFrame({"count": counts, "percent": pct.round(2)}).to_string())
log()
log("       class balance among S1-S4 only:")
log(pd.DataFrame({"count": lab, "percent": lab_pct.round(2)}).to_string())
log(f"  imbalance ratio (largest/smallest) = {lab.max() / lab.min():.1f} : 1")

# ----------------------------------------------------------------------------
# Extra context numbers
# ----------------------------------------------------------------------------
log()
log("=" * 70)
log("ADDITIONAL CONTEXT")
log("=" * 70)
log(f"date range          : {df['creation_time'].min()}  ->  {df['creation_time'].max()}")
log(f"unique products     : {df['product'].nunique():,}")
log(f"unique components   : {df['component'].nunique():,}")
log(f"bugs with dupe_of   : {int(df['dupe_of'].notna().sum()):,}")
log(f"bugs with crash sig : {int(df['has_crash_sig'].sum()):,}")
log()
log("resolution:")
log(df["resolution"].value_counts(dropna=False).to_string())
log()
log("top 10 components:")
log(df["component"].value_counts().head(10).to_string())

(Path(__file__).parent / "stats_summary.txt").write_text(
    "\n".join(log_lines), encoding="utf-8")
print(f"\nSaved 6 figures to {OUT}")
print(f"Saved numbers to {Path(__file__).parent / 'stats_summary.txt'}")
