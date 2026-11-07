# Obesity Level Classification: ML Assignment 2

## a. Problem Statement
Predict a person's **obesity level** (7 classes, from *Insufficient_Weight* to *Obesity_Type_III*) from their eating habits, physical condition and demographic attributes. Five classification algorithms are trained and compared on six evaluation metrics, and the trained models are served through an interactive Streamlit web app.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository, [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) (individuals from Mexico, Peru and Colombia; 77% of records synthetically generated with SMOTE, 23% collected from users).
- **Size:** 2,111 rows (2,087 after removing 24 duplicate rows), **16 features**, 1 target.
- **Target (`NObeyesdad`), 7 classes:** Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III (roughly balanced).

| Feature | Type | Meaning |
|---|---|---|
| Gender | categorical | Male / Female |
| Age, Height, Weight | numeric | Age (years), height (m), weight (kg) |
| family_history_with_overweight | categorical | Family member is or was overweight |
| FAVC | categorical | Frequent consumption of high-calorie food |
| FCVC | numeric | Frequency of vegetable consumption |
| NCP | numeric | Number of main meals per day |
| CAEC | categorical | Eating between meals |
| SMOKE | categorical | Smoker |
| CH2O | numeric | Daily water intake |
| SCC | categorical | Monitors calorie intake |
| FAF | numeric | Physical activity frequency |
| TUE | numeric | Time using technology devices |
| CALC | categorical | Alcohol consumption |
| MTRANS | categorical | Usual transportation mode |

**Pre-processing:** numeric features are standardised (`StandardScaler`) and categorical features are one-hot encoded (`OneHotEncoder`). The pre-processing is bundled with each classifier in a scikit-learn `Pipeline`, so every saved model in `model/` takes raw CSV columns directly. Data split: stratified 80/20 train/test (`random_state=27`). The 418-row test split is saved as `test_data.csv`.

## c. Links
- **GitHub repository:** `https://github.com/csubashGit/ml-assignment-2/`
- **Live Streamlit app:** `https://ml-assignment-2-vnxg4cfqbws2vvnebbfywr.streamlit.app/`

## d. Models Used & Comparison Table
Models: Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine (RBF kernel). AUC is one-vs-rest. Precision, Recall and F1 are weighted averages over the 7 classes.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9330 | 0.9925 | 0.9346 | 0.9330 | 0.9328 | 0.9221 |
| Decision Tree | 0.9211 | 0.9630 | 0.9217 | 0.9211 | 0.9211 | 0.9079 |
| kNN | 0.8373 | 0.9720 | 0.8392 | 0.8373 | 0.8252 | 0.8132 |
| Naive Bayes | 0.4976 | 0.8627 | 0.5115 | 0.4976 | 0.4442 | 0.4352 |
| SVM | **0.9569** | **0.9983** | **0.9570** | **0.9569** | **0.9568** | **0.9497** |

## e. Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall (93.3% accuracy, AUC 0.99). After scaling, the classes are close to linearly separable in the Height/Weight space, so a linear multinomial model already works very well. Most errors are between adjacent classes such as Overweight I and II. |
| Decision Tree | Strong accuracy (92.1%) and easy to interpret, because its splits on Weight and Height approximate BMI thresholds. Its AUC (0.963) is the lowest of the good models because leaf probabilities are coarse, and it overfits more easily than the other models. |
| kNN | Moderate (83.7%). One-hot categorical columns and the distance metric add noise, and neighbours from adjacent obesity levels get mixed up. Its high AUC (0.97) shows the ranking is still good, but hard decisions near class borders suffer. |
| Naive Bayes | Weakest model (49.8% accuracy, MCC 0.44). The feature-independence assumption is badly violated: class membership depends on the interaction of Height and Weight (BMI), and Gaussian likelihoods on one-hot or binary columns are a poor fit. |
| SVM | **Best model on all six metrics** (95.7% accuracy, AUC 0.998, MCC 0.95). The RBF kernel captures the non-linear Height×Weight boundary and the mixed feature interactions. It is the recommended model and the default in the app. |

**Overall:** SVM > Logistic Regression > Decision Tree > kNN >> Naive Bayes. Models that can learn the joint effect of Height and Weight perform best. MCC follows the same ranking as accuracy because the classes are nearly balanced.

## Repository Structure
```
├── app.py              # Streamlit app (loads saved models, never retrains)
├── train_models.py     # Training + evaluation script (creates model/ and test_data.csv)
├── requirements.txt
├── README.md
├── test_data.csv       # Hold-out test set (features + target) to upload in the app
├── data/ObesityDataSet_raw_and_data_sinthetic.csv
└── model/              # 5 saved pipelines (*.joblib) + metrics.csv
```

## How to Run
```bash
pip install -r requirements.txt
python train_models.py      # optional: re-creates model/ and test_data.csv
streamlit run app.py
```
App features: CSV upload, model-selection dropdown, the 6 evaluation metrics, a confusion-matrix heatmap, a classification report, a downloadable sample test file and an all-model comparison tab.
