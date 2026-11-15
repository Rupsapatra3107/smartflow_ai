import numpy as np
import pandas as pd
import pygeohash as pgh
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

CONFIG = dict(
    train_path="data/train.csv",
    test_path="data/test.csv",
    time_col="timestamp",
    lat_col="lat",
    lon_col="lon",
    target="demand",
    geohash_precision=6,
    smoothing=20,  # higher = trust the global mean more for rare geohashes
    seed=42,
)


def add_features(df, c):
    df = df.copy()
    ts = pd.to_datetime(df[c["time_col"]])
    hour = ts.dt.hour + ts.dt.minute / 60
    dow = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["geohash"] = [
        pgh.encode(la, lo, precision=c["geohash_precision"])
        for la, lo in zip(df[c["lat_col"]], df[c["lon_col"]])
    ]
    return df


def bayes_encode(train, target, col, m):
    """Smoothed mean encoding: (n*mean_g + m*global) / (n + m)."""
    g = train.groupby(col)[target].agg(["mean", "count"])
    glob = train[target].mean()
    enc = (g["count"] * g["mean"] + m * glob) / (g["count"] + m)
    return enc, glob


def encode_oof(train, c, n_splits=5):
    """Out-of-fold encoding for train rows so the target doesn't leak."""
    out = pd.Series(index=train.index, dtype=float)
    kf = KFold(n_splits, shuffle=True, random_state=c["seed"])
    for tr, va in kf.split(train):
        enc, glob = bayes_encode(train.iloc[tr], c["target"], "geohash", c["smoothing"])
        out.iloc[va] = train.iloc[va]["geohash"].map(enc).fillna(glob).values
    return out


def main():
    c = CONFIG
    train = add_features(pd.read_csv(c["train_path"]), c)
    test = add_features(pd.read_csv(c["test_path"]), c)

    
    train = train.sort_values(c["time_col"]).reset_index(drop=True)
    cut = int(len(train) * 0.8)
    tr, va = train.iloc[:cut].copy(), train.iloc[cut:].copy()

    tr["geo_enc"] = encode_oof(tr, c)
    enc, glob = bayes_encode(tr, c["target"], "geohash", c["smoothing"])
    va["geo_enc"] = va["geohash"].map(enc).fillna(glob)

    feats = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
             c["lat_col"], c["lon_col"], "geo_enc"]

    lgbm = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03,
                             num_leaves=64, subsample=0.8,
                             colsample_bytree=0.8, random_state=c["seed"])
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                               n_jobs=-1, random_state=c["seed"])
    lgbm.fit(tr[feats], tr[c["target"]])
    rf.fit(tr[feats], tr[c["target"]])

    p_lgb, p_rf = lgbm.predict(va[feats]), rf.predict(va[feats])
    blend = 0.6 * p_lgb + 0.4 * p_rf
    print(f"R2  LightGBM: {r2_score(va[c['target']], p_lgb):.4f}")
    print(f"R2  RandomForest: {r2_score(va[c['target']], p_rf):.4f}")
    print(f"R2  Ensemble: {r2_score(va[c['target']], blend):.4f}")

    full = train.copy()
    full["geo_enc"] = encode_oof(full, c)
    enc, glob = bayes_encode(full, c["target"], "geohash", c["smoothing"])
    test["geo_enc"] = test["geohash"].map(enc).fillna(glob)
    lgbm.fit(full[feats], full[c["target"]])
    rf.fit(full[feats], full[c["target"]])
    test["pred"] = 0.6 * lgbm.predict(test[feats]) + 0.4 * rf.predict(test[feats])
    test["rank"] = test["pred"].rank(ascending=False, method="first").astype(int)
    test.sort_values("rank")[[c["time_col"], "geohash", "pred", "rank"]].to_csv(
        "submission_ranked.csv", index=False)
    print("Wrote submission_ranked.csv")


if __name__ == "__main__":
    main()
