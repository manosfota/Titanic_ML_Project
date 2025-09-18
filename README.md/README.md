# Titanic Survival Prediction

This project is a **Machine Learning pipeline** built with Python to analyze the Titanic dataset and predict passenger survival using **Logistic Regression**.

---

## 🚀 Project Workflow

1. **Data Loading & Inspection**

   - Load Titanic dataset (`train.csv`)
   - Inspect missing values, data types, and summary statistics

2. **Data Cleaning**

   - Handle missing values (`Age`, `Embarked`)
   - Drop irrelevant columns (`Cabin`)
   - Encode categorical variables (`Sex`, `Embarked`)
   - Create new features (`Family_Size`, `Is_Alone`)

3. **Feature Engineering**

   - Extract passenger titles from names
   - Group rare titles together

4. **Visualizations**

   - Survival rate by gender & passenger class
   - Correlation heatmap
   - Age distribution by survival
   - Fare distribution by passenger class

5. **Machine Learning**

   - Feature scaling with `StandardScaler`
   - Train/test split (80/20)
   - Train **Logistic Regression** model

6. **Model Evaluation**

   - Accuracy score
   - Classification report
   - Feature importance analysis

7. **Results Saving**
   - Save cleaned dataset (`cleaned_titanic.csv`)
   - Save predictions with probabilities

---

## 📊 Results

- **Model Accuracy**: ~80%
- **Most important feature**: Passenger sex (`Sex`)
- Predictions saved in a DataFrame with actual survival, predicted survival, and survival probability.

---

## 📂 Project Structure
