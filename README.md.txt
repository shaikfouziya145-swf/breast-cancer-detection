# Breast Cancer Detection using Various ML Algorithms

Classifies breast tumors as **benign** or **malignant** from diagnostic measurements. The project compares six machine learning algorithms and uses cross-validation to find the best performers.

## Dataset
- File: `data.csv`
- 569 samples and 32 columns (ID, diagnosis, and 30 numeric tumor features)
- Target: `diagnosis` (B = Benign → 0, M = Malignant → 1)

## Workflow
1. **Load data** with pandas
2. **Preprocess**: encode the diagnosis as 0/1 and separate features from the target
3. **Split**: 75% training, 25% testing
4. **Scale features** with `StandardScaler`
5. **Compare models** using 10-fold Stratified Cross-Validation
6. **Predict** on the test set using SVM and compare predicted vs. actual classes

## Algorithms compared
| Short name | Algorithm |
|---|---|
| LR | Logistic Regression |
| LDA | Linear Discriminant Analysis |
| KNN | K-Nearest Neighbors |
| CART | Decision Tree |
| NB | Gaussian Naive Bayes |
| SVM | Support Vector Machine |

## Results
Logistic Regression (~98.1%) and SVM (~97.9%) had the highest cross-validation accuracy. A bar chart in the notebook compares all six models.

## Technologies
Python, pandas, NumPy, scikit-learn, Matplotlib

## How to run
1. Download `BreastCancerDetection_VariousMLAlgorithm.ipynb` and `data.csv`
2. Open the notebook in Google Colab
3. Upload `data.csv` when prompted
4. Run all cells