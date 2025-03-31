# Heart Disease Prediction - Hyperparameter Tuning for Recall Optimization

## Project Overview
This project focuses on enhancing the recall of the positive class (heart disease cases, labeled as 1) in the "heart_2020_cleaned.csv" dataset. The primary goal was to maximize the detection of true heart disease cases while balancing false positives.

## Dataset
- **Source:** "heart_2020_cleaned.csv"
- **Preprocessing Steps:**
  - Sampled 50,000 rows from the dataset.
  - Encoded categorical variables using one-hot encoding.
  - Addressed class imbalance using SMOTETomek.
  - Split data into training (80%) and test (20%) sets.

## Methodology
Seven different Random Forest models were trained and evaluated based on recall:
1. **Default Random Forest (Imbalanced, No Resampling)**
2. **Default Random Forest (With SMOTETomek Resampling)**
3. **GridSearchCV Hyperparameter Tuning**
4. **RandomizedSearchCV Hyperparameter Tuning**
5. **Hyperopt Random Search**
6. **Hyperopt Anneal Search**
7. **Scikit-Optimize Hyperparameter Tuning**

## Results
| Model | Recall Score (Test Set) |
|--------|-----------------|
| Default RF (Imbalanced) | 0.1065 |
| Default RF (SMOTETomek) | 0.2805 |
| GridSearchCV | 0.2781 |
| RandomizedSearchCV | 0.4722 |
| Hyperopt Random Search | 0.3704 |
| Hyperopt Anneal Search | 0.3006 |
| Scikit-Optimize | 0.3965 |

## Key Findings
- The **default Random Forest model on imbalanced data** performed the worst, with a recall of **0.1065**.
- Applying **SMOTETomek** significantly improved recall to **0.2805**.
- **RandomizedSearchCV** outperformed all models with a **recall of 0.4722**, making it the best-performing tuning technique.
- **Scikit-Optimize (0.3965)** and **Hyperopt Random Search (0.3704)** were close to the best model while using fewer iterations, making them more computationally efficient.
- **Precision-Recall Curve Analysis:**
  - The optimal threshold for **RandomizedSearchCV** was **0.4131**, achieving a recall of **0.60** and precision of **0.2543**.
  - This threshold optimized recall within a **0.6–0.7 range**, at the cost of increased false positives.
  
## Conclusion
- **RandomizedSearchCV** provided the highest recall but was computationally expensive.
- **Scikit-Optimize** and **Hyperopt** methods achieved competitive results with fewer iterations, making them efficient alternatives.
- The project highlights the importance of hyperparameter tuning and resampling techniques in improving recall for imbalanced datasets.

## Future Work
- Experiment with other boosting models (XGBoost, LightGBM) to compare performance.
- Implement feature selection techniques to improve efficiency.
- Explore automated hyperparameter tuning frameworks.

