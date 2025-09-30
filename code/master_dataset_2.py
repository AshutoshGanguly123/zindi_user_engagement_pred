import pandas as pd
import numpy as np

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "../data"
OUT_PATH = f"{DATA_DIR}/master_dataset_v2.csv"

TOP_N_TITLES = 20        # widen top activity titles
TOP_N_COUNTRIES = 30     # one-hot only the most common countries to avoid explosion
EPS = 1e-9

# ----------------------------
# Load
# ----------------------------
users      = pd.read_csv(f"{DATA_DIR}/Users.csv")
activity   = pd.read_csv(f"{DATA_DIR}/UserActivity.csv")
comp_part  = pd.read_csv(f"{DATA_DIR}/CompetitionPartipation.csv")  # (kept filename as in your code)
discussions= pd.read_csv(f"{DATA_DIR}/Discussion.csv")
comments   = pd.read_csv(f"{DATA_DIR}/Comments.csv")

# minimal key for offset computation
users_key = users[["User_ID", "Created At Year", "Created At Month"]].drop_duplicates()

# ----------------------------
# Helpers
# ----------------------------
def compute_offset(df, user_col, year_col, month_col):
    df_subset = df[[user_col, year_col, month_col]].copy()
    df_subset = df_subset.rename(columns={
        year_col: f"{year_col}_activity",
        month_col: f"{month_col}_activity"
    })
    tmp = df_subset.merge(users_key, on="User_ID", how="left")
    return (tmp[f"{year_col}_activity"] - tmp["Created At Year"]) * 12 + \
           (tmp[f"{month_col}_activity"] - tmp["Created At Month"])

def safe_div(num, den):
    return num / (den.replace(0, np.nan) + EPS)

def entropy_from_counts(counts_row):
    """Shannon entropy over a row of nonnegative counts."""
    vals = counts_row.values.astype(float)
    s = vals.sum()
    if s <= 0:
        return 0.0
    p = vals / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

# ----------------------------
# Month offsets
# ----------------------------
activity["month_offset"]    = compute_offset(activity,   "User_ID", "datetime Year", "datetime Month")
discussions["month_offset"] = compute_offset(discussions,"User_ID", "Created At Year", "Created At Month")
comments["month_offset"]    = compute_offset(comments,   "User_ID", "Created At Year", "Created At Month")
comp_part["month_offset"]   = compute_offset(comp_part,  "User_ID", "Created At Year", "Created At Month")

# ----------------------------
# Target: active in second month (offset == 1)
# ----------------------------
active_in_second = activity.loc[activity["month_offset"] == 1].groupby("User_ID").size()
target = (active_in_second > 0).astype(int).rename("active_second_month")

# ----------------------------
# Filter month 0 slices
# ----------------------------
fm_activity    = activity.loc[activity["month_offset"] == 0].copy()
fm_discussions = discussions.loc[discussions["month_offset"] == 0].copy()
fm_comments    = comments.loc[comments["month_offset"] == 0].copy()
fm_comp        = comp_part.loc[comp_part["month_offset"] == 0].copy()

# ----------------------------
# Base counts (month 0)
# ----------------------------
fm_total = fm_activity.groupby("User_ID").size().rename("fm_total_activities")

# Top titles (global over month 0)
top_titles = (
    fm_activity["Title"]
    .value_counts()
    .head(TOP_N_TITLES)
    .index.tolist()
)

fm_types = (
    fm_activity.loc[fm_activity["Title"].isin(top_titles)]
    .groupby(["User_ID","Title"]).size().unstack(fill_value=0)
)
fm_types.columns = [f"fm_act_{str(c).strip().replace(' ','_')}" for c in fm_types.columns]

# Diversity / distribution over titles
fm_unique_titles = fm_activity.groupby("User_ID")["Title"].nunique().rename("fm_unique_titles")

# Entropy across the same fm_types columns (title mix)
if not fm_types.empty:
    fm_title_entropy = fm_types.apply(entropy_from_counts, axis=1).rename("fm_title_entropy")
else:
    fm_title_entropy = pd.Series(dtype=float, name="fm_title_entropy")

# Dominance: max share among top titles
if not fm_types.empty:
    fm_types_sum = fm_types.sum(axis=1).replace(0, np.nan)
    fm_top_share_max = (fm_types.max(axis=1) / (fm_types_sum + EPS)).fillna(0.0).rename("fm_top_title_max_share")
else:
    fm_top_share_max = pd.Series(dtype=float, name="fm_top_title_max_share")

# Logs (stabilize heavy tails)
fm_total_log1p = np.log1p(fm_total).rename("fm_total_activities_log1p")

# Discussions / comments / competitions
fm_discussions_cnt = fm_discussions.groupby("User_ID").size().rename("fm_discussions")
fm_comments_cnt    = fm_comments.groupby("User_ID").size().rename("fm_comments")
fm_comp_cnt        = fm_comp.groupby("User_ID").size().rename("fm_comp_participations")

# Competition diversity (unique competitions)
if "Competition_ID" in fm_comp.columns:
    fm_comp_unique = fm_comp.groupby("User_ID")["Competition_ID"].nunique().rename("fm_comp_unique")
else:
    fm_comp_unique = pd.Series(dtype=float, name="fm_comp_unique")

# Combined text interactions + ratios
fm_text_interactions = (fm_discussions_cnt.reindex(fm_total.index, fill_value=0) + 
                        fm_comments_cnt.reindex(fm_total.index, fill_value=0)).rename("fm_text_interactions")

fm_comments_ratio   = safe_div(fm_comments_cnt.reindex(fm_total.index, fill_value=0), fm_total).rename("fm_comments_ratio")
fm_disc_ratio       = safe_div(fm_discussions_cnt.reindex(fm_total.index, fill_value=0), fm_total).rename("fm_disc_ratio")
fm_comp_ratio       = safe_div(fm_comp_cnt.reindex(fm_total.index, fill_value=0), fm_total).rename("fm_comp_ratio")
fm_text_int_ratio   = safe_div(fm_text_interactions, fm_total).rename("fm_text_interactions_ratio")

# Binary flags
fm_any_discussion = (fm_discussions_cnt > 0).astype(int).rename("fm_any_discussion")
fm_any_comment    = (fm_comments_cnt > 0).astype(int).rename("fm_any_comment")
fm_any_comp       = (fm_comp_cnt > 0).astype(int).rename("fm_any_comp")

# Interaction features
fm_total_x_unique = (fm_total * fm_unique_titles.reindex(fm_total.index, fill_value=0)).rename("fm_total_x_unique_titles")
fm_text_x_total   = (fm_text_interactions * fm_total).rename("fm_text_interactions_x_total")

# Proportions of each top title among total (normalized composition)
if not fm_types.empty:
    fm_types_prop = fm_types.div(fm_total, axis=0).fillna(0.0)
    fm_types_prop.columns = [c + "_prop" for c in fm_types_prop.columns]
else:
    fm_types_prop = pd.DataFrame(index=fm_total.index)

# ----------------------------
# User-side features
# ----------------------------
# Keep your original user columns; add signup month dummies and cyclical encoding
base_user_cols = ["FeatureX", "FeatureY", "Countries_ID", "Created At Month"]
for col in base_user_cols:
    if col not in users.columns:
        users[col] = np.nan

# Signup month (as provided)
users["signup_month"] = users["Created At Month"].astype(int)

# One-hot for signup month (seasonality)
signup_month_dummies = pd.get_dummies(users["signup_month"], prefix="signup_month")

# Cyclical encoding for signup month (0..11 assumed)
users["signup_month_sin"] = np.sin(2 * np.pi * (users["signup_month"] % 12) / 12.0)
users["signup_month_cos"] = np.cos(2 * np.pi * (users["signup_month"] % 12) / 12.0)

# Countries one-hot for top N countries (avoid huge dimensionality)
top_countries = users["Countries_ID"].value_counts().head(TOP_N_COUNTRIES).index
users["Countries_ID_trim"] = np.where(users["Countries_ID"].isin(top_countries), users["Countries_ID"], "OTHER")
countries_dummies = pd.get_dummies(users["Countries_ID_trim"], prefix="country")

# ----------------------------
# Assemble master
# ----------------------------
user_indexed = users.set_index("User_ID")[base_user_cols + ["signup_month_sin", "signup_month_cos"]].copy()

to_join = [
    fm_total, fm_total_log1p,
    fm_discussions_cnt, fm_comments_cnt, fm_comp_cnt,
    fm_comp_unique,
    fm_unique_titles, fm_title_entropy, fm_top_share_max,
    fm_text_interactions,
    fm_comments_ratio, fm_disc_ratio, fm_comp_ratio, fm_text_int_ratio,
    fm_any_discussion, fm_any_comment, fm_any_comp,
    fm_total_x_unique, fm_text_x_total
]

# Join dense frames
master = (
    user_indexed
    .join(to_join, how="left")
    .join(fm_types, how="left")
    .join(fm_types_prop, how="left")
    .join(signup_month_dummies.set_index(users["User_ID"]), how="left")
    .join(countries_dummies.set_index(users["User_ID"]), how="left")
    .join(target, how="left")
)

# Fill & types
master = master.fillna({"active_second_month": 0}).fillna(0).reset_index()
master["active_second_month"] = master["active_second_month"].astype(int)

# ----------------------------
# Save & quick summary
# ----------------------------
master.to_csv(OUT_PATH, index=False)
print("SAVED:", OUT_PATH)
print("SHAPE:", master.shape)
print("LABEL_DIST:", master["active_second_month"].value_counts(dropna=False).to_dict())

# Optional: quick sanity peek at the most important engineered cols
cols_preview = [
    "fm_total_activities","fm_total_activities_log1p",
    "fm_unique_titles","fm_title_entropy","fm_top_title_max_share",
    "fm_discussions","fm_comments","fm_comp_participations","fm_comp_unique",
    "fm_text_interactions","fm_comments_ratio","fm_disc_ratio","fm_comp_ratio","fm_text_interactions_ratio",
    "fm_any_discussion","fm_any_comment","fm_any_comp",
    "signup_month_sin","signup_month_cos"
]
print("PREVIEW_COLS_PRESENT:", [c for c in cols_preview if c in master.columns])
