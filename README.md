Project Overview

This project aims to predict an employee’s salary and typical workhours using machine learning techniques. It serves as a decision-support tool for HR professionals and job seekers to estimate compensation trends and workload expectations based on experience and job designation.

Objectives
* Predict salary based on experience and designation.
* Estimate typical workhours per week for a given role.
* Analyze compensation patterns and workload trends across different job titles.
* Provide an interactive and user-friendly tool for quick predictions.

Key Features
* Dual Prediction Models:
    * Workhours Prediction: Estimates typical weekly hours based on experience and designation.
    * Salary Prediction: Predicts salary using experience, designation, and predicted workhours.
* Data Preprocessing:
    * Categorical encoding using LabelEncoder.
    * Handling of numerical and categorical data for accurate modeling.
* User-Friendly Interface:
    * Input only experience and designation.
    * Receive predicted workhours and salary instantly.
* Optional Data Visualization:
    * Salary distribution across roles.
    * Experience vs Salary scatterplots.

Technology Stack
* Python – Programming language.
* Pandas – Data manipulation and preprocessing.
* NumPy – Numerical operations.
* Scikit-learn – KNN regression and model evaluation.
* Matplotlib & Seaborn – Data visualization (optional).

Dataset Description
* Dataset Name: Employee Salary Dataset
* Features: Name, Age, Designation, Experience, Workhours, Salary
* Target Variables:
    * Salary – Predicted based on role and experience.
    * Workhours – Predicted typical workhours based on role and experience.
 
Machine Learning Approach
* K-Nearest Neighbors (KNN) Regressor for both workhours and salary prediction.
* Training & Testing: 80/20 split for accurate evaluation.
* Evaluation Metrics:
    * Mean Squared Error (MSE)
    * R2 Score

Expected Outcomes
* Accurate salary predictions for different roles and experience levels.
* Typical weekly workhours estimation to understand workload expectations.
* Insights into compensation trends across various job designations.

Future Enhancements
* Add feature importance analysis to identify key factors influencing salary.
* Implement a web application using Flask or Streamlit for real-time predictions.
* Integrate additional features such as location, industry, and certifications for more accurate predictions.
* Explore advanced regression models like Random Forest or Gradient Boosting for improved performance.
