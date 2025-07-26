# split_data.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

# 1) Load data
df = pd.read_csv("data/Reviews.csv")

# 2) Helper columns
df["word_count"] = df["Text"].fillna("").str.split().apply(len)
df["Time_dt"] = pd.to_datetime(df["Time"], unit="s")

# 3) Helpfulness Heuristic (HH)
den = df["HelpfulnessDenominator"].replace(0, np.nan)
df["help_ratio"] = (df["HelpfulnessNumerator"] / den).fillna(0)
hh_genuine = (df["HelpfulnessDenominator"] >= 5) & (df["help_ratio"] >= 0.6)
hh_fake    = (df["HelpfulnessDenominator"] >= 5) & (df["help_ratio"] <= 0.1) & (df["word_count"] < 30)
df["hh_label"] = np.where(hh_genuine, 1,
                   np.where(hh_fake, 0, np.nan))

# 4) Sentiment Proxy (SP)
df["sp_label"] = np.where(df["Score"].isin([4,5]), 1,
                   np.where(df["Score"].isin([1,2]), 0, np.nan))

# 5) Burst/Duplicate Heuristic (BD)
def compute_window(grp):
    # Convert timestamps to integer seconds and numpy array
    times = (grp["Time_dt"].astype(np.int64) // 1_000_000_000).to_numpy()
    products = grp["ProductId"].to_numpy()
    n = len(times)
    reviews_24h = np.zeros(n, dtype=int)
    uniq_24h    = np.zeros(n, dtype=int)
    start = 0
    for end in range(n):
        # slide window start until within 24h
        while times[end] - times[start] > 86400:
            start += 1
        window = slice(start, end+1)
        reviews_24h[end] = end - start + 1
        uniq_24h[end]    = len(np.unique(products[window]))
    out = grp.copy()
    out["reviews_24h"]        = reviews_24h
    out["unique_products_24h"] = uniq_24h
    return out

df = df.sort_values(["UserId", "Time_dt"])
df = df.groupby("UserId", group_keys=False).apply(compute_window)

bd_fake   = (df["reviews_24h"] >= 3) & (df["unique_products_24h"] >= 3)
user_max  = df.groupby("UserId")["reviews_24h"].max()
genuine_u = user_max[user_max <= 1].index
df["bd_label"] = np.where(df["UserId"].isin(genuine_u), 1,
                    np.where(bd_fake, 0, np.nan))

# 6) Create splits directory
out_dir = Path("project_splits")
out_dir.mkdir(exist_ok=True)

# 7) Split & save helper
def split_and_save(df, label_col, prefix):
    sub = df.dropna(subset=[label_col]).copy()
    sub["label"] = sub[label_col].astype(int)
    # 70% train, 30% temp
    train, temp = train_test_split(
        sub, test_size=0.3, stratify=sub["label"], random_state=SEED
    )
    # Of the 30% temp, split 1/3 val (10% total), 2/3 test (20% total)
    val, test = train_test_split(
        temp, test_size=2/3, stratify=temp["label"], random_state=SEED
    )
    # Save
    train.to_csv(out_dir / f"{prefix}_train.csv", index=False)
    val.to_csv(out_dir   / f"{prefix}_val.csv",   index=False)
    test.to_csv(out_dir  / f"{prefix}_test.csv",  index=False)
    print(f"{prefix.upper()}: train={len(train):,}, val={len(val):,}, test={len(test):,}")

# 8) Generate for each task
split_and_save(df, "hh_label", "hh")
split_and_save(df, "sp_label", "sp")
split_and_save(df, "bd_label", "bd")

print("✅ All splits saved in", out_dir.resolve())
