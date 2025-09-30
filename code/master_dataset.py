import pandas as pd
import numpy as np

# Load
users = pd.read_csv("../data/Users.csv")
activity = pd.read_csv("../data/UserActivity.csv") 
comp_part = pd.read_csv("../data/CompetitionPartipation.csv")
discussions = pd.read_csv("../data/Discussion.csv")
comments = pd.read_csv("../data/Comments.csv")

users_key = users[["User_ID", "Created At Year", "Created At Month"]].drop_duplicates()

def compute_offset(df, user_col, year_col, month_col):
    # Rename columns to avoid conflicts
    df_subset = df[[user_col, year_col, month_col]].copy()
    df_subset = df_subset.rename(columns={
        year_col: f"{year_col}_activity",
        month_col: f"{month_col}_activity"
    })
    
    tmp = df_subset.merge(users_key, on="User_ID", how="left")
    return (tmp[f"{year_col}_activity"] - tmp["Created At Year"]) * 12 + \
           (tmp[f"{month_col}_activity"] - tmp["Created At Month"])

activity["month_offset"] = compute_offset(activity, "User_ID", "datetime Year", "datetime Month")
discussions["month_offset"] = compute_offset(discussions, "User_ID", "Created At Year", "Created At Month")
comments["month_offset"] = compute_offset(comments, "User_ID", "Created At Year", "Created At Month")
comp_part["month_offset"] = compute_offset(comp_part, "User_ID", "Created At Year", "Created At Month")

# Target
active_in_second = activity.loc[activity["month_offset"] == 1].groupby("User_ID").size()
target = (active_in_second > 0).astype(int).rename("active_second_month")

# Month 0 features
fm_activity = activity.loc[activity["month_offset"] == 0]
fm_total = fm_activity.groupby("User_ID").size().rename("fm_total_activities")

TOP_N = 15
top_titles = activity["Title"].value_counts().head(TOP_N).index.tolist()
fm_types = (
    fm_activity.loc[fm_activity["Title"].isin(top_titles)]
    .groupby(["User_ID","Title"]).size().unstack(fill_value=0)
)
fm_types.columns = [f"fm_act_{c}".replace(" ","_") for c in fm_types.columns]

fm_discussions = discussions.loc[discussions["month_offset"] == 0].groupby("User_ID").size().rename("fm_discussions")
fm_comments = comments.loc[comments["month_offset"] == 0].groupby("User_ID").size().rename("fm_comments")
fm_comp_parts = comp_part.loc[comp_part["month_offset"] == 0].groupby("User_ID").size().rename("fm_comp_participations")

master = (
    users.set_index("User_ID")[["FeatureX","FeatureY","Countries_ID","Created At Month"]]
    .join([fm_total, fm_discussions, fm_comments, fm_comp_parts, fm_types], how="left")
    .join(target, how="left")
    .fillna({"active_second_month":0})
    .fillna(0)
    .reset_index()
    .astype({"active_second_month":int})
)

# Save and preview
out_path = "../data/master_dataset.csv"
master.to_csv(out_path, index=False)


print("SAVED", out_path)
print("SHAPE", master.shape)
print("LABEL_DIST", master["active_second_month"].value_counts().to_dict())
