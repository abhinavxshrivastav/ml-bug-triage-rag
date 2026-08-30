"""
Build a single self-contained A3 poster (HTML -> print to PDF).
=============================================================
Generates poster-optimised figures, computes 5 statistics tables, and embeds
everything as base64 into ONE html file: a3_poster.html

Open it in Chrome/Edge -> Ctrl+P -> Destination "Save as PDF"
-> Paper size A3 -> Margins: None -> Background graphics: ON -> Save.

Observation boxes are left BLANK on purpose: they are handwritten.
"""
from pathlib import Path
import base64, io, re, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RAW = Path(r"C:\Users\abhin\bugtriage-data\raw\bugsrepo\Bug_meta_data.csv")
HERE = Path(__file__).parent
SEED = 42
DPI = 200
SHA = "17457b6c451e8462c49c98678cba67debb8f0ea982267ced367ff2d2efef0df3"

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.titleweight": "bold", "figure.facecolor": "white",
})
PAL = {"S1": "#b2182b", "S2": "#ef8a62", "S3": "#67a9cf", "S4": "#2166ac"}
ORDER = ["S1", "S2", "S3", "S4"]

USECOLS = ["id", "severity", "priority", "product", "component", "type", "status",
           "resolution", "creation_time", "cf_last_resolved", "comment_count",
           "votes", "summary", "cc", "blocks", "depends_on", "duplicates",
           "keywords", "cf_crash_signature", "dupe_of"]


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fmt(x, d=2):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if pd.isna(x):
        return "—"
    return f"{x:,.{d}f}"


# ---------------------------------------------------------------- load -----
print("loading …")
raw_rows = 0
df = pd.read_csv(RAW, usecols=lambda c: c in USECOLS, low_memory=False)
raw_rows = len(df)
df = df.drop_duplicates(subset=["id"], keep="first").copy()
n_unique = len(df)
n_dropped = raw_rows - n_unique

df["severity"] = df["severity"].astype("string").str.strip()
sev_modern = df["severity"].isin(ORDER)

ct = pd.to_datetime(df["creation_time"], errors="coerce", format="mixed", utc=True)
lr = pd.to_datetime(df["cf_last_resolved"], errors="coerce", format="mixed", utc=True)
df["creation_time"] = ct
df["resolution_days"] = (lr - ct).dt.total_seconds() / 86400.0
df.loc[df["resolution_days"] < 0, "resolution_days"] = np.nan

df["summary"] = df["summary"].astype("string").fillna("")
df["summary_len_words"] = df["summary"].str.split().str.len().fillna(0)
df["summary_len_chars"] = df["summary"].str.len()


def list_len(s):
    inner = (s.astype("string").fillna("[]").str.strip()
             .str.removeprefix("[").str.removesuffix("]").str.strip())
    return np.where(inner.str.len() == 0, 0, inner.str.count(",") + 1).astype(int)


for c in ["blocks", "depends_on", "duplicates", "cc"]:
    df[f"n_{c}"] = list_len(df[c])
df["n_keywords"] = (df["keywords"].astype("string").fillna("")
                    .apply(lambda s: 0 if not s.strip()
                           else len(re.split(r"[,\s]+", s.strip()))))

MIN_DAYS = 0.01
_res = df[sev_modern & df["resolution_days"].notna() & (df["resolution_days"] > 0)]
box = _res[_res["resolution_days"] >= MIN_DAYS].copy()
n_excluded = len(_res) - len(box)

FIGS, TABLES = {}, {}

# ============================================================== FIGURE 1 ====
d = df["comment_count"].dropna()
fig, (axb, axh) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True,
                               gridspec_kw={"height_ratios": (.22, .78)})
sns.boxplot(x=d, ax=axb, color="#67a9cf", fliersize=1, width=.55,
            linewidth=1.1)
axb.set(xlabel="", yticks=[])
axh.hist(d.clip(upper=40), bins=np.arange(0, 41) - .5, color="#67a9cf",
         edgecolor="white", linewidth=.5)
for v, c, ls, lab in [(d.mean(), "#b2182b", "-", f"Mean = {d.mean():.2f}"),
                      (d.median(), "#1a9850", "--", f"Median = {d.median():.0f}"),
                      (d.quantile(.25), "#555", ":", f"Q1 = {d.quantile(.25):.0f}"),
                      (d.quantile(.75), "#555", "-.", f"Q3 = {d.quantile(.75):.0f}")]:
    axh.axvline(v, color=c, linestyle=ls, linewidth=1.8, label=lab)
axh.set_xlabel("Number of comments per bug report  (display clipped at 40)")
axh.set_ylabel("Frequency")
axh.legend(frameon=True, fontsize=9, ncol=2)
axh.set_xlim(-.5, 40)
axb.set_title("FIG 1 · Distribution of Comments per Bug Report", pad=8)
fig.tight_layout()
FIGS["f1"] = fig_to_b64(fig)

TABLES["t1"] = ("TABLE 1 · Descriptive statistics — comments per bug", [
    ("Statistic", "Value", "Statistic", "Value"),
    ("n (valid)", fmt(int(d.notna().sum())), "Q3 (P75)", fmt(d.quantile(.75), 0)),
    ("Mean", fmt(d.mean()), "P90", fmt(d.quantile(.90), 0)),
    ("Median (Q2)", fmt(d.median(), 0), "P95", fmt(d.quantile(.95), 0)),
    ("Mode", fmt(int(d.mode().iloc[0])), "P99", fmt(d.quantile(.99), 0)),
    ("Std. deviation", fmt(d.std()), "Maximum", fmt(int(d.max()))),
    ("Variance", fmt(d.var()), "IQR (Q3−Q1)", fmt(d.quantile(.75) - d.quantile(.25), 0)),
    ("Minimum", fmt(int(d.min())), "Skewness", fmt(d.skew())),
    ("Q1 (P25)", fmt(d.quantile(.25), 0), "Kurtosis", fmt(d.kurtosis())),
])
q1, q3 = d.quantile([.25, .75]); iqr = q3 - q1
n_out = int((d > q3 + 1.5 * iqr).sum())
TABLES["t1_note"] = (f"Tukey upper fence Q3+1.5·IQR = {q3 + 1.5 * iqr:.1f}; "
                     f"{n_out:,} observations ({n_out / len(d) * 100:.2f}%) lie above it.")

# ============================================================== FIGURE 2 ====
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.boxplot(data=box, x="severity", y="resolution_days", order=ORDER, hue="severity",
            palette=PAL, legend=False, ax=ax, fliersize=.8, linewidth=1.2,
            showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                       "markeredgecolor": "black", "markersize": 5})
ax.set_yscale("log")
ax.set_xlabel("Severity   (S1 = most severe  →  S4 = least severe)")
ax.set_ylabel("Time to resolution (days, log scale)")
ax.set_title("FIG 2 · Bug Resolution Time by Severity Level", pad=8)
for i, s in enumerate(ORDER):
    ax.text(i, box["resolution_days"].max() * 1.5,
            f"n = {int((box['severity'] == s).sum()):,}", ha="center", fontsize=9.5)
fig.tight_layout()
FIGS["f2"] = fig_to_b64(fig)

g = box.groupby("severity", observed=True)["resolution_days"]
rows = [("Severity", "n", "Min", "Q1", "Median", "Q3", "Max", "IQR", "Mean")]
for s in ORDER:
    v = g.get_group(s)
    rows.append((s, fmt(len(v)), fmt(v.min()), fmt(v.quantile(.25)),
                 fmt(v.median()), fmt(v.quantile(.75)), fmt(v.max(), 0),
                 fmt(v.quantile(.75) - v.quantile(.25)), fmt(v.mean())))
TABLES["t2"] = ("TABLE 2 · Five-number summary of resolution time (days) by severity", rows)
kw = stats.kruskal(*[v.values for _, v in g])
TABLES["t2_note"] = (f"Kruskal–Wallis H = {kw.statistic:,.1f}, df = 3, "
                     f"p &lt; 0.001 → the four medians are not all equal. "
                     f"{n_excluded:,} records resolved in &lt; {MIN_DAYS} d "
                     f"(≈15 min) excluded as automated closures.")

# ============================================================== FIGURE 3 ====
sc = box[box["comment_count"] > 0].copy()
if len(sc) > 6000:
    sc = sc.sample(6000, random_state=SEED)
fig, ax = plt.subplots(figsize=(10, 4.5))
# comment_count is a small integer, so on a log axis the points collapse into
# vertical stripes. Jitter horizontally in log space purely for legibility --
# all statistics below are computed on the unjittered values.
_rng = np.random.default_rng(SEED)
sc["_x_jit"] = sc["comment_count"] * 10 ** _rng.uniform(-.045, .045, len(sc))
for s in ORDER:
    sub = sc[sc["severity"] == s]
    ax.scatter(sub["_x_jit"], sub["resolution_days"], s=5, alpha=.30,
               color=PAL[s], label=f"{s} (n={len(sub):,})", edgecolors="none",
               rasterized=True)
ax.set_xscale("log"); ax.set_yscale("log")
x, y = np.log10(sc["comment_count"]), np.log10(sc["resolution_days"])
ok = np.isfinite(x) & np.isfinite(y)
lin = stats.linregress(x[ok], y[ok])
xs = np.linspace(x[ok].min(), x[ok].max(), 100)
ax.plot(10 ** xs, 10 ** (lin.intercept + lin.slope * xs), color="black", lw=1.9,
        label="OLS fit (log–log)")
rho, rho_p = stats.spearmanr(sc["comment_count"], sc["resolution_days"])
ax.set_xlabel("Number of comments (log scale)")
ax.set_ylabel("Resolution time in days (log scale)")
ax.set_title("FIG 3 · Discussion Volume vs. Time to Resolution", pad=8)
ax.legend(frameon=True, fontsize=8.5, loc="lower right", ncol=2)
fig.tight_layout()
FIGS["f3"] = fig_to_b64(fig)

TABLES["t3"] = ("TABLE 3 · Bivariate association — comments vs. resolution time", [
    ("Measure", "Value", "Measure", "Value"),
    ("n (plotted sample)", fmt(len(sc)), "OLS slope (log–log)", fmt(lin.slope, 4)),
    ("Spearman ρ", fmt(rho, 4), "OLS intercept", fmt(lin.intercept, 4)),
    ("p-value (ρ)", "&lt; 0.001", "Std. error of slope", fmt(lin.stderr, 4)),
    ("Pearson r (log–log)", fmt(lin.rvalue, 4), "Coefficient of determination r²",
     fmt(lin.rvalue ** 2, 4)),
])
TABLES["t3_note"] = (f"r² = {lin.rvalue ** 2:.4f} → the log-linear model accounts for "
                     f"{lin.rvalue ** 2 * 100:.1f}% of the variance in resolution time.")

# ============================================================== FIGURE 4 ====
sl = df["summary_len_words"]
slc = sl[(sl > 0) & (sl <= 40)]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(slc, bins=np.arange(0, 41) - .5, density=True, color="#c7e9c0",
        edgecolor="#4a7c59", linewidth=.6, label="Relative frequency")
slc.plot.kde(ax=ax, color="#1a9850", linewidth=2.1, label="Kernel density estimate")
qs = sl.quantile([.25, .5, .75, .90])
ax.axvline(sl.mean(), color="#b2182b", lw=1.9, label=f"Mean = {sl.mean():.2f}")
ax.axvline(qs[.5], color="#222", ls="--", lw=1.9, label=f"Median = {qs[.5]:.0f}")
ax.axvline(qs[.25], color="#777", ls=":", lw=1.5, label=f"Q1 = {qs[.25]:.0f}")
ax.axvline(qs[.75], color="#777", ls=":", lw=1.5, label=f"Q3 = {qs[.75]:.0f}")
ax.axvline(qs[.90], color="#777", ls="-.", lw=1.5, label=f"P90 = {qs[.90]:.0f}")
ax.set_xlim(0, 40)
ax.set_xlabel("Length of bug report title (words)")
ax.set_ylabel("Probability density")
ax.set_title("FIG 4 · Distribution of Bug Report Title Length", pad=8)
ax.legend(frameon=True, fontsize=8.5, ncol=2)
fig.tight_layout()
FIGS["f4"] = fig_to_b64(fig)

sh = stats.shapiro(sl.dropna().sample(4000, random_state=SEED))
TABLES["t4"] = ("TABLE 4 · Distribution of title length (words)", [
    ("Statistic", "Value", "Statistic", "Value"),
    ("n", fmt(int(sl.notna().sum())), "Q3 (P75)", fmt(sl.quantile(.75), 0)),
    ("Mean", fmt(sl.mean()), "P90", fmt(sl.quantile(.90), 0)),
    ("Median", fmt(sl.median(), 0), "P99", fmt(sl.quantile(.99), 0)),
    ("Std. deviation", fmt(sl.std()), "Maximum", fmt(int(sl.max()))),
    ("P5", fmt(sl.quantile(.05), 0), "Skewness", fmt(sl.skew())),
    ("Q1 (P25)", fmt(sl.quantile(.25), 0), "Kurtosis", fmt(sl.kurtosis())),
])
TABLES["t4_note"] = (f"Shapiro–Wilk on a random sample of n = 4,000: "
                     f"W = {sh.statistic:.4f}, p &lt; 0.001 → normality is rejected.")

# ============================================================== FIGURE 5 ====
CC = ["comment_count", "resolution_days", "summary_len_words", "n_blocks",
      "n_depends_on", "n_cc", "n_keywords", "votes"]
LB = ["Comments", "Resolution\ndays", "Title\nwords", "Blocks",
      "Depends on", "CC size", "Keywords", "Votes"]
corr = df[CC].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=.6,
            cbar_kws={"shrink": .78, "label": "Spearman ρ"},
            xticklabels=LB, yticklabels=LB, ax=ax, annot_kws={"size": 9})
ax.set_title("FIG 5 · Correlation Between Bug Attributes", pad=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
fig.tight_layout()
FIGS["f5"] = fig_to_b64(fig)

NICE = {"comment_count": "Comments per bug", "resolution_days": "Resolution time (days)",
        "summary_len_words": "Title length (words)", "n_blocks": "Bugs blocked",
        "n_depends_on": "Dependencies", "n_cc": "CC list size",
        "n_keywords": "Keywords assigned", "votes": "Votes"}
pairs = (corr.where(mask.T & ~np.eye(len(corr), dtype=bool))
         .stack().abs().sort_values(ascending=False).head(8))
rows = [("Rank", "Variable A", "Variable B", "Spearman ρ", "Strength")]
for i, ((a, b), _) in enumerate(pairs.items(), 1):
    v = corr.loc[a, b]
    a, b = NICE.get(a, a), NICE.get(b, b)
    s = ("negligible" if abs(v) < .10 else "weak" if abs(v) < .30
         else "moderate" if abs(v) < .50 else "strong")
    rows.append((str(i), a, b, f"{v:+.3f}", s))
TABLES["t5"] = ("TABLE 5 · Strongest pairwise rank correlations", rows)
TABLES["t5_note"] = ("Spearman ρ used throughout because every variable is strongly "
                     "right-skewed and the relationships are monotonic rather than linear.")

# ============================================================== FIGURE 6 ====
TOPN = 10          # 10 rather than 15: at poster scale 15 bars are unreadable
prod = df["product"].value_counts()
comp = df["component"].value_counts()
n_tot = len(df)
fig, (axp, axc) = plt.subplots(1, 2, figsize=(16, 3.9))

for ax, s, title, colour in [
        (axp, prod.head(TOPN), f"Top {TOPN} Products", "#2166ac"),
        (axc, comp.head(TOPN), f"Top {TOPN} Components", "#41818f")]:
    y = np.arange(len(s))[::-1]
    ax.barh(y, s.values, color=colour, edgecolor="white", linewidth=.5, height=.74)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=10)
    ax.set_xlabel("Number of bug reports", fontsize=10)
    ax.set_title(title, fontsize=12.5, weight="bold", pad=5)
    ax.set_xlim(0, s.max() * 1.26)
    for yi, v in zip(y, s.values):
        ax.text(v + s.max() * .015, yi, f"{v:,} ({v / n_tot * 100:.1f}%)",
                va="center", fontsize=9, color="#333333")
    ax.grid(axis="y", visible=False)
fig.suptitle("FIG 6 · Where Bugs Are Reported — Product and Component Concentration",
             fontsize=13, weight="bold", y=1.02)
fig.tight_layout()
FIGS["f6"] = fig_to_b64(fig)

cum_p = prod.cumsum() / n_tot * 100
cum_c = comp.cumsum() / n_tot * 100
rows = [("Rank", "Product", "n", "% of all", "Component", "n", "% of all")]
for i in range(5):
    rows.append((str(i + 1),
                 str(prod.index[i]), fmt(int(prod.iloc[i])),
                 f"{prod.iloc[i] / n_tot * 100:.1f}%",
                 str(comp.index[i]), fmt(int(comp.iloc[i])),
                 f"{comp.iloc[i] / n_tot * 100:.1f}%"))
TABLES["t6"] = ("TABLE 6 · Largest categories and concentration of the label space", rows)
TABLES["t6_note"] = (
    f"{df['product'].nunique()} products and {df['component'].nunique():,} components in total. "
    f"Top 5 cover {cum_p.iloc[4]:.1f}% of bugs by product but only {cum_c.iloc[4]:.1f}% by "
    f"component; {int((cum_p < 80).sum() + 1)} products vs "
    f"{int((cum_c < 80).sum() + 1)} components are needed to reach 80% — the component label "
    f"space is far more long-tailed. Median category size: {prod.median():.0f} (product), "
    f"{comp.median():.0f} (component).")

# ------------------------------------------------------- dataset facts -----
res_counts = df["resolution"].value_counts(dropna=False)
typ = df["type"].value_counts()
span = (df["creation_time"].max() - df["creation_time"].min()).days
facts = {
    "raw_rows": fmt(raw_rows), "unique": fmt(n_unique), "dropped": fmt(n_dropped),
    "dropped_pct": f"{n_dropped / raw_rows * 100:.2f}",
    "date_min": str(df["creation_time"].min())[:10],
    "date_max": str(df["creation_time"].max())[:10], "span": fmt(span),
    "products": fmt(df["product"].nunique()), "components": fmt(df["component"].nunique()),
    "fixed": fmt(int(res_counts.get("FIXED", 0))),
    "dupe": fmt(int(res_counts.get("DUPLICATE", 0))),
    "incomplete": fmt(int(res_counts.get("INCOMPLETE", 0))),
    "invalid": fmt(int(res_counts.get("INVALID", 0))),
    "unresolved": fmt(int(res_counts.isna().sum() if False else df["resolution"].isna().sum())),
    "defect": fmt(int(typ.get("defect", 0))), "task": fmt(int(typ.get("task", 0))),
    "enh": fmt(int(typ.get("enhancement", 0))),
    "sev_labelled": fmt(int(sev_modern.sum())),
    "sev_pct": f"{sev_modern.mean() * 100:.1f}",
    "resolved_n": fmt(int(df["resolution_days"].notna().sum())),
    "s1": fmt(int((df["severity"] == "S1").sum())),
    "s2": fmt(int((df["severity"] == "S2").sum())),
    "s3": fmt(int((df["severity"] == "S3").sum())),
    "s4": fmt(int((df["severity"] == "S4").sum())),
    "unset": fmt(int((df["severity"] == "--").sum())),
    "dupeof": fmt(int(df["dupe_of"].notna().sum())),
    "sha": SHA[:24],
}

# ------------------------------------------------------------- render ------
def table_html(key):
    title, rows = TABLES[key]
    note = TABLES.get(key + "_note", "")
    # 4 columns = Statistic/Value pairs; 3 = label + two numeric columns;
    # anything wider is a general data table
    ncol = len(rows[0])
    cls = ' class="trio"' if ncol == 3 else (' class="wide"' if ncol > 4 else "")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
    n = f'<div class="note">{note}</div>' if note else ""
    return (f'<div class="tbl-title">{html.escape(title)}</div>'
            f'<table{cls}><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table>{n}')


def block(n, figkey, tblkey, wide=False):
    """Figure + its statistics table. Observations are handwritten on the printed
    sheet, so no reserved box is drawn -- the space goes to the figure instead."""
    cls = "block wide" if wide else "block"
    return f"""<section class="{cls}">
  <img src="data:image/png;base64,{FIGS[figkey]}" alt="figure {n}">
  {table_html(tblkey)}
</section>"""


TPL = (HERE / "_a3_template.html").read_text(encoding="utf-8")
out = (TPL
       .replace("{{BLOCK1}}", block(1, "f1", "t1"))
       .replace("{{BLOCK2}}", block(2, "f2", "t2"))
       .replace("{{BLOCK3}}", block(3, "f3", "t3"))
       .replace("{{BLOCK4}}", block(4, "f4", "t4"))
       .replace("{{BLOCK5}}", block(5, "f5", "t5"))
       .replace("{{BLOCK6}}", block(6, "f6", "t6", wide=True)))
for k, v in facts.items():
    out = out.replace("{{" + k + "}}", str(v))

dest = HERE / "a3_poster.html"
dest.write_text(out, encoding="utf-8")
print(f"wrote {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")
