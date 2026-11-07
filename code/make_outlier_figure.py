import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
o = pd.read_csv("reports/modeling_v3/oof_residuals_with_prior.csv")
thr = 0.6617516
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
ax[0].hist(o.log_resid, bins=80, color="#8aa1b8"); ax[0].axvline(thr, color="#c0392b", ls="--"); ax[0].axvline(-thr, color="#c0392b", ls="--")
ax[0].set_xlabel("Log residual (actual minus predicted)"); ax[0].set_ylabel("Player-seasons")
ax[0].set_title("Out-of-fold residuals; dashed lines = 2 scaled MAD (0.66)", fontsize=10)
c = np.where(o.is_outlier, "#c0392b", "#8aa1b8"); la, lp = np.log(o.salary), np.log(o.pred)
ax[1].scatter(lp, la, s=5, c=c, alpha=.6, linewidths=0)
lim = [min(la.min(), lp.min()), max(la.max(), lp.max())]; ax[1].plot(lim, lim, color="k", lw=.8)
ax[1].set_xlabel("Predicted log salary"); ax[1].set_ylabel("Recorded log salary")
ax[1].set_title("Flagged rows (red) are 16.5% of the sample", fontsize=10)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("reports/modeling_v3/fig_outliers.png")
