# Comparing Classifiers

## Predicting Term Deposit Subscriptions Using Classification Models

### Business Understanding
- The goal of this project is to help a Portuguese bank predict whether a client will subscribe to a term deposit based on historical direct marketing campaign data.
- Since marketing campaigns can be expensive and intrusive, the bank seeks a data-driven model that enables targeted outreach to likely responders.
- The dataset is highly imbalanced (~88% "no", ~12% "yes"), so a good model must go beyond accuracy and focus on identifying true positives (clients who will say yes).

### Approach and Methodology
The CRISP-DM methodology was used:
1. **Data Understanding:**
    - Used the [UCI's Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (41,188 rows).
    - Focused on bank client features and excluded high-leakage columns like duration.
2. **Feature Engineering:**
    - Categorical features were one-hot encoded.
    - The target (y) was mapped to binary (0 = no, 1 = yes). 
    - Class imbalance was handled using class_weight='balanced'.
3. **Modeling & Evaluation:**
    - Models used:
      - Logistic Regression 
      - K-Nearest Neighbors (KNN)
      - Decision Tree 
      - Support Vector Machine (SVM)
    - Metrics Used:
      - F1-score (class 1)
      - Recall (class 1)
      - ROC-AUC 
      - Accuracy (used cautiously due to imbalance)
    - Hyperparameter tuning via GridSearchCV.

    
#### Key Findings

| Model                           | Accuracy | Recall (Class 1) | F1-score (Class 1) | ROC-AUC |
| ------------------------------- | -------- | ---------------- | ------------------ | ------- |
| Logistic Regression (balanced)  | 0.5846   | 0.62             | 0.25               | 0.65    |
| Linear SVM (balanced)           | 0.6069   | 0.58             | 0.25               | N/A     |
| KNN (tuned)                     | 0.8594   | 0.13             | 0.17               | 0.5678  |
| Decision Tree (tuned, balanced) | 0.6480   | 0.54             | **0.26**           | 0.6381  |

- Decision Tree (with class_weight='balanced') achieved the highest F1-score for the minority class(1), offering the best balance of precision and recall.
- Logistic Regression and Linear SVM performed similarly, with slightly higher recall but lower F1.
- KNN and non-tuned Decision Trees performed well on accuracy but failed to detect meaningful positives.

#### Recommendation
Based on model performance:
- Recommended Model: Tuned Decision Tree with class_weight='balanced'
  - Best F1-score (class 1)
  - Interpretable and fast 
  - Suitable for production deployment to prioritize outreach
- Alternatives: Logistic Regression or Linear SVM if model simplicity or inference speed is prioritized.

#### Project Files
| File                                                             | Description                                                |
|------------------------------------------------------------------|------------------------------------------------------------|
| [bhaiml_pract_app_03.ipynb](notebooks/bhaiml_pract_app_03.ipynb) | CRISP-DM analysis and training & comparison of classifiers |
| [bank-additional-full.csv](data/bank-additional-full.csv)        | Original dataset                                           |

