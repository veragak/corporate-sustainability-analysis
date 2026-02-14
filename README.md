# ESG Classification & Environmental Clustering  
**Machine Learning Analysis of Corporate Sustainability Performance**

This project develops a data-driven framework to analyze and predict corporate ESG performance using environmental, financial, and firm-level characteristics.  
It combines unsupervised clustering with supervised classification models and explainable AI techniques to study sustainability patterns across companies.

Developed for **FEM11149 – Introduction to Data Science**  
Erasmus School of Economics, Erasmus University Rotterdam

---

## Repository Structure
- data/
    - `company_esg_financial_dataset.csv` # Input dataset (if available)
- code/
    - analysis_script.R # Full, reproducible analysis pipeline
- report/
    - esg_analysis.pdf # Final academic report
    - esg_analysis.Rmd
- README.md


---

## Overview

The objective of this project is to understand how environmental and financial variables relate to corporate ESG performance and to assess the predictive power of different machine learning models.

The analysis addresses the following questions:

- Can companies be grouped into meaningful environmental sustainability clusters?
- Which financial and environmental variables best predict ESG performance?
- How do interpretable models compare to more complex ensemble methods?
- How robust are predictions across different model specifications?
- Which variables drive global and local ESG predictions?

---

## Methods

### Data Preparation
- Removal of company identifiers
- Factor encoding of categorical variables (industry, region, year)
- Median imputation for missing growth rates
- Log-transformation of skewed environmental variables
- Binary ESG classification target:
  - **High ESG** vs **Low ESG** (median split)

---

### Unsupervised Learning: Environmental Clustering
- Hierarchical clustering (Ward.D2)
- Euclidean distance on scaled environmental variables:
  - Carbon emissions
  - Water usage
  - Energy consumption
- Cluster profiling and labeling:
  - Sustainable
  - Moderate impact
  - High impact

---

### Supervised Learning: ESG Classification Models

Models estimated and compared:

- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Tuned Random Forest (OOB-optimized `mtry`)

Predictors include:
- Financial performance (revenue, profit margin, growth rate, market cap)
- Environmental indicators
- Industry, region, and year effects

Evaluation metrics:
- Accuracy
- Sensitivity & specificity
- Balanced accuracy
- Cohen’s Kappa
- Confusion matrix visualization

---

### Model Interpretation

**Global interpretation**
- Random Forest variable importance (Mean Decrease Gini)
- Partial dependence plots for key predictors
- Regional comparison of predicted ESG probabilities

**Local interpretation**
- LIME explanations for:
  - Most confident High ESG prediction
  - Most confident Low ESG prediction
- Feature-level contributions to individual predictions

---

## Key Findings

- Environmental variables play a major role in ESG classification, alongside firm size and profitability.
- Random Forest models outperform linear and tree-based alternatives on test accuracy and balanced accuracy.
- Model performance improves with hyperparameter tuning.
- Interpretability tools reveal:
  - Strong nonlinear effects
  - Regional differences in ESG likelihood
  - Distinct drivers behind High vs Low ESG classifications
- Ensemble models offer superior predictive power, while interpretable models remain valuable for transparency.

---

## Reproducing the Analysis

### Install required packages

```r
install.packages(c(
  "tidyverse", "ggplot2", "ggrepel", "patchwork", "caret",
  "knitr", "kableExtra", "rpart", "rpart.plot",
  "randomForest", "pdp", "lime", "scales"
))
```

### Run the analysis
- Place the dataset in: `data/company_esg_financial_dataset.csv`
- Run: `source("code/analysis.R")`

---

## Skills Demonstrated
- Data cleaning and feature engineering
- Hierarchical clustering and cluster interpretation
- Logistic regression and tree-based models
- Random Forest tuning and evaluation
- Model comparison using multiple performance metrics
- Explainable AI (PDP & LIME)
- Reproducible research workflow in R
- Academic reporting and visualization

---

## Author

Vera Gak Anagrova
MSc Data Science & Marketing Analytics
Erasmus School of Economics
Erasmus University Rotterdam
