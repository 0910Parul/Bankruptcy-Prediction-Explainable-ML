# Interpretable Machine Learning for Corporate Bankruptcy Prediction

This project predicts which companies will go bankrupt from their financial statements, and, more importantly, **explains why**. Using **132 manufacturing and retail firms** (66 bankrupt) described by **24 accounting ratios**, I tuned and validated a decision tree and a random forest (**ROC-AUC 0.883**). I then interpreted the models with four methods: impurity importance, permutation importance, TreeSHAP and KernelSHAP. Finally, I turned local SHAP explanations into **two distinct distress profiles** with tailored client recommendations. Built for **MSBA 6341 — Responsible AI** (Carlson, University of Minnesota).

## 🔍 Project Goals

- Build a bankruptcy prediction model with sound training and evaluation practices on a very small dataset.
- Explain how the model makes predictions, and test whether different interpretation methods tell the same story.
- Translate model explanations into recommendations that executives and individual at-risk clients can act on.

## 📄 Files Included

- **`Bankruptcy_Interpretability_Analysis.ipynb`**: full analysis, covering model tuning and evaluation, four global interpretation methods, rank-agreement analysis, and SHAP-based client segmentation.
- **`report/Bankruptcy_Interpretability_Report.pdf`**: final written report with answers and charts.
- **`docs/Case_Brief.pdf`**: the case prompt and the definitions of all 24 ratios.
- **`bankruptcy.csv`**: dataset (132 firms × 24 ratios; `D = 1` healthy, `D = 0` failed).
- **`requirements.txt`**: Python dependencies.

## 🧪 Methods Used

- **Model training and validation:** 80/20 train-test split, then **GridSearchCV with 5-fold stratified cross-validation** to tune tree depth, leaf size and forest size, so the small sample doesn't lead to overfitting.
- **Evaluation:** precision, recall, F1, accuracy, **ROC-AUC** and confusion matrices, with explicit reporting of recall on failed firms (the costly error).
- **Global interpretation:** Gini importance, **permutation importance** (AUC drop), **TreeSHAP** and model-agnostic **KernelSHAP**, plus a direct reading of the decision tree's splits.
- **Method-agreement analysis:** a **Spearman rank correlation** between the four importance rankings, and a consensus ranking for executives.
- **Local explanations to segments:** per-firm SHAP profiles of model-flagged firms, clustered with **k-means** (k chosen by silhouette score) and visualized with PCA.

## 📈 Key Findings

- The **random forest outperformed the decision tree** on the held-out test set, with ROC-AUC **0.883 vs. 0.852**, accuracy **81.5% vs. 77.8%**, and higher recall on failed firms (77.8% vs. 72.2%). Cross-validated F1 was **0.855**.
- **The interpretation methods agree, except for the single tree.** Random-forest importance, TreeSHAP and KernelSHAP rankings correlate at **0.96–0.98 (Spearman)**, and all three rank **ASSETS/DEBTS** first. The decision tree correlates at only **0.17–0.23**, because with correlated ratios a single tree gives most of the credit to whichever feature it splits on.
- **Firm health comes down to solvency and debt coverage.** The consensus top signals are ASSETS/DEBTS, income-plus-depreciation/DEBTS, INC/DEBTS, INC/ASSETS and WCFO/DEBTS.
- The model flagged **62 of 132 firms** as at risk, and **92% of those actually failed**.
- Clustering their SHAP profiles revealed **two distress patterns**:
  - **Profitability / cash-flow distress** (45 firms): median INC/ASSETS of −0.04, vs. +0.06 for healthy firms.
  - **Leverage / liquidity distress** (17 firms): ASSETS/DEBTS of ~1.4 vs. 2.1, and current ratio of ~1.5 vs. 2.2.

## 🛠 Technologies

- Python (pandas, NumPy, scikit-learn, SciPy)
- SHAP (TreeExplainer, KernelExplainer)
- Matplotlib
- Jupyter Notebook

## ✅ Outcome

The project delivers an executive-ready view of **what drives bankruptcy risk**, built on the signals that every reliable interpretation method agrees on, together with **segment-specific playbooks**:
- **Profitability / cash-flow segment:** margin, cost and cash-generation fixes.
- **Leverage / liquidity segment:** debt restructuring and working-capital fixes.
- **Both segments:** an early-warning dashboard tracking assets-to-debt and cash-flow-to-debt.

All recommendations are framed as associations to investigate with expert review, not proven causal levers.

---

By **Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parulchaud) · [Email](mailto:parul.jaswant@gmail.com)
