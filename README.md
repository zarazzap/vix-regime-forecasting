# VIX Volatility Regime Forecasting

**Module:** MSIN0097 Predictive Analytics (UCL, 2025–26)

## Project Objective

This project investigates whether machine-learning models trained on historical VIX features can improve next-day volatility regime prediction beyond a strong persistence baseline. The VIX (Cboe Volatility Index) is formulated as a three-class classification problem with regimes defined as Low (VIX ≤ 15), Medium (15 < VIX ≤ 25), and High (VIX > 25).

## Summary

The analysis demonstrates that VIX regime persistence is so strong (lag-1 autocorrelation > 0.98) that no ML model achieves a statistically significant improvement over simply carrying forward today's regime. Logistic regression achieves the highest point estimate (macro F1 = 0.8821 vs persistence 0.8799), but a McNemar test on 28 discordant pairs yields p = 0.85. The project provides a rigorous null result: VIX-only features cannot overcome the informational ceiling set by VIX's nature as a forward-looking, options-implied measure.

## Dataset

- **Source:** CBOE VIX Historical Data (`VIX_History.csv`)
- **Period:** January 1990 – February 2026 (9,132 trading days)
- **Fields:** DATE, OPEN, HIGH, LOW, CLOSE
- **After feature construction:** 9,110 labelled observations
- **Class distribution:** Low 32.5%, Medium 50.2%, High 17.3%

## Repository Structure

```
├── analysis_v3.ipynb          # Main analysis notebook (end-to-end pipeline)
├── VIX_History.csv            # Raw VIX daily OHLC data
├── VIX_predictive_analytics_final_report.docx  # Final submission report
├── Predictive Analytics report.docx            # Draft report (reference)
├── MSIN0097_ Predictive Analytics 25-26 Individual Coursework.pdf  # Assessment brief
├── MSIN0097_Marking_Feedback_1.docx            # Examiner feedback on draft
├── screenshots/               # Agent interaction evidence & notebook outputs
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
└── README.md                  # This file
```

## Methodology

The project follows a standard end-to-end ML workflow:

1. **Problem Framing:** One-step-ahead three-class classification with macro F1 as primary metric.
2. **EDA:** Distribution analysis, autocorrelation, regime transition matrix, volatility clustering.
3. **Feature Engineering:** Nine leakage-free features derived from VIX closes (lagged values, log returns, rolling statistics, mean-reversion signal, current regime encoding).
4. **Data Preparation:** Chronological 70/15/15 split; StandardScaler fitted on training set only.
5. **Modelling:** Five ML models compared against persistence and dummy baselines.
6. **Evaluation:** Macro F1, per-class F1, McNemar test, failure-mode analysis on transition days.

## Models

| Model | Architecture / Config | Test Macro F1 |
|-------|----------------------|---------------|
| Persistence baseline | Carry forward today's regime | 0.8799 |
| Logistic Regression | C=10.0, lbfgs solver | 0.8821 |
| Random Forest | 300 trees, no max depth | 0.8677 |
| XGBoost | lr=0.05, max_depth=5, 300 trees | 0.8663 |
| sklearn MLP | (64,), alpha=0.0001, early stopping | 0.8756 |
| PyTorch DeepMLP | 128→64→32, Dropout 0.20, Adam | 0.8787 |

## Key Findings

- **Persistence dominates:** The naive carry-forward baseline (macro F1 = 0.8799) is not statistically exceeded by any model (McNemar p = 0.85 for best model).
- **Regime persistence is structural:** Transition-matrix diagonal entries of 0.897–0.927 mean regime changes occur on only ~7–11% of days.
- **Linear sufficiency:** The predictive signal is approximately linear; logistic regression marginally outperforms deeper architectures.
- **Transition-day weakness:** All models struggle on the 130 regime transitions in the test set (error rates 18–25% vs 7–12% on stable days).
- **VIX-only ceiling:** The options-implied nature of VIX sets a fundamental information ceiling that single-series features cannot overcome.

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Execution

1. Place `VIX_History.csv` in the same directory as the notebook.
2. Open `analysis_v3.ipynb` in Jupyter Notebook or JupyterLab.
3. Run **Kernel → Restart & Run All** for a clean end-to-end execution.

The notebook is self-contained: it loads data, engineers features, trains all models, and produces all evaluation outputs including figures and performance tables.

## Dependencies

See `requirements.txt`. Key packages:
- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- xgboost
- torch (PyTorch)
- matplotlib

## Limitations

- All features are univariate (VIX-only); cross-asset signals (equity returns, credit spreads, realised volatility) may improve transition-day prediction.
- Fixed regime thresholds (15/25) are applied across the full sample period despite structural shifts in VIX levels.
- The validation set (2015–2020) contains the March 2020 COVID spike, which may bias hyperparameter selection toward tail-event robustness.

## Agent Tooling

This project used Claude Code as an AI coding assistant for scaffolding, debugging, and documentation. All agent contributions were verified through kernel restarts, leakage checks, and end-to-end reruns. See Appendix A of the report for the full Agent Usage Log and Decision Register.
