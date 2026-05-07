import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

RNG_SEED = 42
np.random.seed(RNG_SEED)

def main(csv_path, limit_rows=-1):
    # Load data
    df = pd.read_csv(csv_path, usecols=["pc", "delta_in", "delta_out"], engine="c", low_memory=False)
    if limit_rows > 0:
        df = df.iloc[:limit_rows].copy()
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    df["delta_in"] = (df["delta_in"]).astype(np.int64)
    df["delta_out"] = (df["delta_out"]).astype(np.int64)
    df["pc_id"] = df["pc"].astype("category").cat.codes.astype(np.int32)
    
    # Split: 60% train, 10% val, 30% test
    n = len(df)
    n_train = int(n * 0.60)
    n_val = int(n * 0.10)
    
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    
    features = ["pc_id", "delta_in"]
    X_train, y_train = train[features], train["delta_out"]
    X_val, y_val = val[features], val["delta_out"]
    X_test, y_test = test[features], test["delta_out"]
    
    # Stage 1: Binary classifier (zero vs nonzero)
    y_train_cls = (y_train != 0).astype(int)
    y_val_cls = (y_val != 0).astype(int)
    
    clf = lgb.LGBMClassifier(
        class_weight='balanced', learning_rate=0.1, n_estimators=400,
        subsample=0.9, colsample_bytree=0.9, min_data_in_leaf=50,
        random_state=RNG_SEED, verbose=-1
    )
    clf.fit(X_train, y_train_cls, eval_set=[(X_val, y_val_cls)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    cls_pred = clf.predict(X_test)
    
    # Stage 2: Regressor for nonzero predictions
    nz_train_mask = (y_train != 0).values
    X_train_nz = X_train[nz_train_mask]
    y_train_nz = y_train[nz_train_mask]
    
    nz_val_mask = (y_val != 0).values
    X_val_nz = X_val[nz_val_mask]
    y_val_nz = y_val[nz_val_mask]
    
    reg_nz = lgb.LGBMRegressor(
        subsample=0.9, colsample_bytree=0.9, min_data_in_leaf=50,
        learning_rate=0.1, n_estimators=400, max_depth=-1,
        random_state=RNG_SEED, verbose=-1
    )
    reg_nz.fit(X_train_nz, y_train_nz, eval_set=[(X_val_nz, y_val_nz)],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    # Combine predictions
    combined_pred = np.zeros(len(y_test), dtype=float)
    nz_test_idx = np.where(cls_pred == 1)[0]
    if len(nz_test_idx) > 0:
        combined_pred[nz_test_idx] = reg_nz.predict(X_test.iloc[nz_test_idx])
    
    # Final accuracy: exact match after rounding
    y_pred_rounded = np.rint(combined_pred).astype(int)
    accuracy = (y_pred_rounded == y_test.values.astype(int)).mean()
    
    print(f"\n=== RESULTS ===")
    print(f"Hit Rate (Exact Match Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--limit_rows", type=int, default=-1, help="Limit rows (-1 for all)")
    args = ap.parse_args()
    main(args.csv, args.limit_rows)