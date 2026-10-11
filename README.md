# Predicting Olympic Medals: Milano Cortina 2026

**1st place, Canisius International Data Science Masters Challenge (Olympiad) 2026.** 48-hour team competition, judged by U.S. and Canadian industry professionals.

**My role:** Modeling Lead for a 4-person team. I built the predictive model (Task 3): feature engineering, class imbalance handling, model comparison, threshold tuning, and SHAP explanations.

**Result:** XGBoost classifier with **AUC-ROC 0.848**, beating Random Forest (0.779) and a "predict no medal for everyone" baseline that scores 73% accuracy while finding zero medalists.

---

## The question

USA won 33 medals at Milano Cortina 2026, Canada won 21. But who actually outperformed expectations given their resources, and what should each country do differently for 2030? My part was the modeling piece: can we predict which athletes win a medal from their physical, training, and country data?

## The data

- 387 athletes (after cleaning), 23 nations, 14 winter sports, 17 original variables
- Provided by the competition and intentionally messy: planted impossible values, hidden duplicates, and missing data that was not random (reaction time and World Cup points were missing by sport)
- Only about 27% of athletes won a medal, so the classes are imbalanced

## What I did

1. **Feature engineering.** Added 3 features on top of the cleaned data: `Experience_Score`, `Aerobic_Efficiency`, and `Training_Durability`, plus flags for the two variables that were missing by sport. 18 features in total.
2. **Split first, then everything else.** 80/20 stratified train/test split before any preprocessing or class weight calculation, so nothing from the test set leaked into training.
3. **Class imbalance.** Used class weights plus threshold tuning instead of SMOTE. With only 83 medalists in training, synthetic athletes risk creating impossible physiological profiles, and tree models handle class weights well.
4. **Two models.** Random Forest and XGBoost, compared on precision, recall, F1, and AUC-ROC instead of accuracy.
5. **Threshold tuning.** Tested decision thresholds from 0.20 to 0.70 to balance catching medalists against false alarms.
6. **Explainability.** SHAP plots for the 7 athletes named in the brief, showing which features pushed each prediction up or down.

## Results

Test set: 78 athletes, 21 medalists.

| Model | Threshold | Precision (medal) | Recall (medal) | F1 (medal) | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest | 0.50 | 0.50 | 0.38 | 0.43 | 0.779 |
| Random Forest | 0.35 | 0.53 | 0.76 | 0.63 | 0.779 |
| XGBoost | 0.50 | 0.83 | 0.48 | 0.61 | 0.848 |
| **XGBoost** | **0.30** | **0.67** | **0.67** | **0.67** | **0.848** |

**Why not accuracy:** 73% of athletes didn't medal, so a model that always says "no medal" gets 73% accuracy and is useless. Missing a future medalist is the costly mistake here, so recall, F1, and AUC matter more.

**What drove predictions:** VO2max ranked first, followed by sport, winter sport tradition, and country GDP. The 3 engineered features together made up about 18% of total feature importance.

![SHAP explanation for one of the 7 athletes](Task_3_Predictive_Modelling/shap_1_Jordan_Stolz.png)

## Limitations and what I'd do next

- **The threshold was tuned on the test set.** That makes the 0.30 threshold numbers optimistic. The default threshold results and AUC-ROC are clean. Next time I'd tune the threshold with cross-validation on the training data and keep the test set fully untouched.
- **Small test set.** 78 athletes means a couple of predictions can move the metrics a lot. Cross-validated scores would give a more stable estimate.
- **Competition dataset.** It uses real athlete names but was built for the challenge with planted errors, so the model shows the method, not a real Olympic forecasting tool.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
cd Task_3_Predictive_Modelling
jupyter notebook Task_3_NN.ipynb
```

The notebook reads `../data/milan_cortina_2026_cleaned_Final.csv`.

## Repo structure

```
Task_1_Data_Cleanup_Final.R        Data cleaning (R)
Data_Cleaning_Report.pdf           Cleaning decisions and before/after summary
Task2.pdf                          Exploratory analysis and visuals
Task_3_Predictive_Modelling/       Modeling notebook and SHAP plots (my work)
Task3_Modeling_Report.docx         Full modeling write-up (my work)
Task_4_Executive_Summary.pdf       Executive summary for decision makers
Task_Business_ Insight_Report_Team_ 2.pdf   Business insight report
data/                              Cleaned datasets
```

## Team

Team 2, Canisius University. Roles were split by task: data cleaning (Task 1), exploratory analysis and storytelling (Task 2), modeling (Task 3, me), and the business insight report (Task 4).

<details>
<summary>Original challenge brief</summary>

Brief provided by the organizers: use athlete data from Milano Cortina 2026 (390 records, 17 variables) to compare USA and Canada's performance relative to resources, and recommend what each should do for 2030.

Four tasks: (1) data cleaning and preprocessing, (2) exploratory analysis and storytelling, (3) predictive modeling (medal vs no medal, at least 2 models, explicit class imbalance handling, evaluated on accuracy, precision, recall, F1 and AUC-ROC, applied to 7 named athletes), and (4) an executive business insight report.

</details>
