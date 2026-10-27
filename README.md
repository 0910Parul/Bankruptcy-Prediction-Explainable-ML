# Predicting Corporate Bankruptcy with Explainable Machine Learning

## Project Overview

This project combines bankruptcy prediction with model interpretation to support financial-risk analysis. It evaluates decision tree and random forest classifiers, compares explanations of their predictions, and explores how local SHAP profiles can inform different review priorities for at-risk firms.

The dataset contains **132 manufacturing and retail firms**, including **66 failed firms**, described by **24 accounting ratios**. The label convention is **`D = 1` for healthy firms and `D = 0` for failed firms**.

## Key Highlights

- **Predictive comparison:** the random forest recorded **0.883 test ROC-AUC** and **81.5% accuracy**, versus 0.852 and 77.8% for the decision tree.
- **Failure detection:** the random forest identified **14 of 18 failed firms** in the held-out test set, for **77.8% failed-firm recall**.
- **Explainability:** compared impurity importance, permutation importance, TreeSHAP, and KernelSHAP.
- **Decision support:** explored two SHAP-based distress profiles with different profitability, cash-flow, leverage, and liquidity patterns.

Read the [analysis notebook](Bankruptcy_Interpretability_Analysis.ipynb), [report](report/Bankruptcy_Interpretability_Report.pdf), and [results evidence](docs/RESULTS_EVIDENCE.md).

## Technical Implementation

1. **Partition the data:** use a fixed 80/20 train-test split (`random_state=42`), resulting in 105 training firms and 27 test firms. The initial holdout split is not stratified.
2. **Tune models:** use GridSearchCV with 5-fold stratified cross-validation on the training data. Tune tree depth and minimum leaf size, plus forest size.
3. **Evaluate:** report class-level precision, recall, F1, accuracy, ROC-AUC, and confusion matrices. Explicitly distinguish the healthy positive class from failed firms.
4. **Explain predictions:** inspect tree splits, impurity importance, permutation AUC drops, TreeSHAP, and KernelSHAP.
5. **Compare interpretations:** calculate Spearman rank correlations and a consensus ranking across forest impurity importance, TreeSHAP, and KernelSHAP.
6. **Explore segments:** score all firms, cluster local SHAP profiles of model-flagged firms using standardized features and k-means, and inspect silhouette scores and PCA visualizations.

## Methods & Technologies

- **Python analysis:** pandas, NumPy, scikit-learn, SciPy, and Matplotlib.
- **Models and validation:** DecisionTreeClassifier, RandomForestClassifier, GridSearchCV, and StratifiedKFold.
- **Interpretability:** permutation importance, SHAP TreeExplainer, SHAP KernelExplainer, and rank-agreement analysis.
- **Segmentation:** StandardScaler, KMeans, silhouette score, and PCA.
- **Environment:** Jupyter Notebook; dependencies in [requirements.txt](requirements.txt).

## Results & Insights

All model metrics below come from saved outputs on the **27-firm held-out test set**: 18 failed firms and 9 healthy firms. Training was not rerun for this documentation update.

| Model | Test ROC-AUC | Test accuracy | Recall: failed firms (`D=0`) | F1: healthy firms (`D=1`) |
|---|---:|---:|---:|---:|
| Decision tree | 0.852 | 77.8% | 72.2% | 0.727 |
| Random forest | **0.883** | **81.5%** | **77.8%** | **0.762** |

The selected forest used **200 trees, maximum depth 3, and minimum leaf size 1**. Its best mean cross-validation F1 was **0.855 for the healthy class**; this is distinct from held-out test F1 and failure-detection performance.

### Interpretation and exploratory segmentation

- **Debt coverage was prominent:** ASSETS/DEBTS ranked first in forest impurity importance, TreeSHAP, and KernelSHAP. Other leading signals included INCDEP/DEBTS, INC/DEBTS, INC/ASSETS, and WCFO/DEBTS.
- **Explanations showed both agreement and variation:** the three forest-based rankings correlated at **0.96–0.98**, while the decision-tree ranking correlated at **0.17–0.23** with them. Permutation importance also highlighted current liquidity, but many estimates were near zero.
- **Full-sample exploration:** the trained forest flagged **62 of 132 firms**, of which **57 had failed** (about 92%). Because this analysis includes training firms, that percentage is an exploratory fit summary, not an out-of-sample precision estimate.
- **Two tentative profiles:** k-means separated 45 firms with stronger profitability/cash-flow distress signals and 17 with leverage/liquidity signals. The selected **silhouette score was 0.246**, indicating limited separation.

The small test set and correlated financial ratios limit the stability of both performance estimates and feature attributions. Model explanations describe associations; they do not establish that changing a ratio will prevent bankruptcy.

## Repository Structure

| Path | Purpose |
|---|---|
| [Bankruptcy_Interpretability_Analysis.ipynb](Bankruptcy_Interpretability_Analysis.ipynb) | Training, evaluation, interpretation, ranking comparisons, and segmentation |
| [bankruptcy.csv](bankruptcy.csv) | 132 firms, 24 ratios, and the `D` label |
| [docs/Case_Brief.pdf](docs/Case_Brief.pdf) | Case context and ratio definitions |
| [report/Bankruptcy_Interpretability_Report.pdf](report/Bankruptcy_Interpretability_Report.pdf) | Full written analysis and figures |
| [requirements.txt](requirements.txt) | Python dependencies |
| [docs/RESULTS_EVIDENCE.md](docs/RESULTS_EVIDENCE.md) | Metric provenance and evaluation boundaries |

## Getting Started

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook Bankruptcy_Interpretability_Analysis.ipynb
```

Run the notebook in order with the repository root as the working directory so `bankruptcy.csv` resolves correctly. KernelSHAP can take longer than the tree-model steps. Dependencies are unpinned, and SHAP output formats may vary across versions; the notebook contains handling for the saved run's class-output shape.

The saved outputs and report provide the historical results. A fresh run is needed to establish reproducibility in a new environment.

## Applications & Takeaways

This analysis illustrates how to turn model outputs into reviewable risk signals. An analyst can compare the cost of missed failures, identify recurring debt-coverage signals, and present different questions for firms with cash-generation problems versus leverage or liquidity concerns.

The proposed playbooks focus on reviewing margins and cash generation for the first profile, and debt structure and working capital for the second. These are analytical priorities for expert review; no intervention or realized financial benefit was tested. Future validation would require new firms, a suitable failure-focused evaluation objective, and assessment of the segment stability.

## Project Context

Developed for **MSBA 6341 — Responsible AI**, Carlson School of Management, University of Minnesota. Original analysis, data, report, and class-label conventions are retained.

---

By **Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parulchaud) · [Email](mailto:parul.jaswant@gmail.com)
