# Results Evidence and Evaluation Boundaries

Sources are the [analysis notebook](../Bankruptcy_Interpretability_Analysis.ipynb), [CSV](../bankruptcy.csv), and [report](../report/Bankruptcy_Interpretability_Report.pdf). Cell positions below are **1-based**, including markdown. Saved results were inspected, not rerun.

| Claim | Evidence |
|---|---|
| Dataset and split | CSV; notebook cell 4: 105 training rows and 27 test rows, 24 features; `train_test_split` has no `stratify` argument |
| Label meaning | Notebook cell 5: `D=1` healthy, `D=0` failed; test composition 18 failed / 9 healthy |
| Cross-validation | Cell 6: 5-fold shuffled stratified CV; cells 8 and 14 use `scoring="f1"`, optimizing the healthy class |
| Decision tree | Cell 9: ROC-AUC 0.852; accuracy 0.778; failed-firm recall 0.722; class-1 F1 0.727; confusion matrix `[[13,5],[1,8]]` |
| Random forest | Cells 14–15: CV F1 0.855; test ROC-AUC 0.883; accuracy 0.815; failed-firm recall 0.778; class-1 F1 0.762; confusion matrix `[[14,4],[1,8]]` |
| Selected forest parameters | Cell 14: 200 trees, depth 3, minimum leaf size 1 |
| Explanation agreement | Cells 24–25: RF/TreeSHAP/KernelSHAP correlations 0.96–0.98; DT correlations 0.17–0.23 |
| Consensus signals | Cell 27: R21, R18, R15, R14, R24 lead the average ranking |
| Full-sample flagged firms | Cell 29: 62 flagged of 132, 57 failed; 92% is rounded and includes training cases |
| Segmentation | Cells 29–30: silhouette 0.246 at k=2 versus 0.164 at k=3 and 0.148 at k=4; groups of 45 and 17 |

## Interpretation boundaries

- Accuracy and ROC-AUC refer to the held-out 27-firm set; mean tuning CV F1 and full-sample flag precision are separate quantities.
- The holdout split is not stratified, although cross-validation is. Cross-validation reduces dependence on a single training split but does not eliminate overfitting or quantify deployment performance.
- The exploratory SHAP segmentation scores all 132 firms, including training observations. The 92% flag summary must not be presented as prospective detection precision.
- The silhouette score supports tentative profiles in this sample, rather than strongly separated or externally validated client segments.
- The consensus combines forest impurity, TreeSHAP, and KernelSHAP rankings. Permutation importance is a complementary analysis and is not included in that consensus or the four-column Spearman comparison.
- Financial ratios are correlated, which can affect all feature-attribution methods. Agreement between methods is useful evidence about the fitted model, not proof of causal drivers.

No executable files, datasets, reports, or saved outputs are changed by this documentation update.
