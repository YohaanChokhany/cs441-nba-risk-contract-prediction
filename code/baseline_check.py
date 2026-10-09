import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1])
ID = ["salary_row_id","player_id","player_name","season","team_abbr","salary_source","match_stage","in_injury_window","fold","time_holdout","salary","log_salary","salary_pct_cap"]
F = [c for c in d.columns if c not in ID]
X = d[F].astype(float).values
def ridge_fit_predict(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    ym = ytr.mean(); w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym
r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
for tgt in ["log_salary", "salary_pct_cap"]:
    y = d[tgt].values; oof = np.zeros(len(d))
    for k in range(5):
        te = d.fold.values == k
        oof[te] = ridge_fit_predict(X[~te], y[~te], X[te])
    pred_usd = np.exp(oof) if tgt == "log_salary" else oof * d.salary.values / d.salary_pct_cap.values
    mae = np.abs(pred_usd - d.salary.values).mean()
    # time holdout: train before 2023, test 2023-2025
    te = d.time_holdout.values == 1
    p = ridge_fit_predict(X[~te], y[~te], X[te])
    pu = np.exp(p) if tgt == "log_salary" else p * (d.salary.values / d.salary_pct_cap.values)[te]
    print(f"{tgt}: grouped 5-fold R2 {r2(y, oof):.3f}, MAE {mae/1e6:.2f}M dollars | time holdout (2023-2025) R2 {r2(y[te], p):.3f}, MAE {np.abs(pu - d.salary.values[te]).mean()/1e6:.2f}M dollars")
# leakage guard: confirm no feature contains the salary itself
c = pd.DataFrame(X, columns=F).corrwith(d.log_salary).abs().sort_values(ascending=False)
print("max |r| with log salary:", c.head(3).round(3).to_dict())
