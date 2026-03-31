import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load Dataset
data = pd.read_csv('employee_details.csv')


# Encode Designation
label_encoder = LabelEncoder()
data['Designation'] = label_encoder.fit_transform(data['Designation'])

# Train KNN model to predict Salary (Experience + Designation + Workhours)
X_salary = data[['Experience', 'Designation', 'Workhours']]
y_salary = data['Salary']

X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(
    X_salary, y_salary, test_size=0.2, random_state=42
)

knn_salary = KNeighborsRegressor(n_neighbors=3)
knn_salary.fit(X_train_sal, y_train_sal)

# Evaluate Salary model
y_pred_sal = knn_salary.predict(X_test_sal)
print("Salary Model - Mean Squared Error:", mean_squared_error(y_test_sal, y_pred_sal))
print("Salary Model - R2 Score:", r2_score(y_test_sal, y_pred_sal))


# Train KNN model to predict Workhours (Experience + Designation)
X_workhours = data[['Experience', 'Designation']]
y_workhours = data['Workhours']

X_train_wh, X_test_wh, y_train_wh, y_test_wh = train_test_split(
    X_workhours, y_workhours, test_size=0.2, random_state=42
)

knn_workhours = KNeighborsRegressor(n_neighbors=3)
knn_workhours.fit(X_train_wh, y_train_wh)

# Evaluate Workhours model
y_pred_wh = knn_workhours.predict(X_test_wh)
print("Workhours Model - Mean Squared Error:", mean_squared_error(y_test_wh, y_pred_wh))
print("Workhours Model - R2 Score:", r2_score(y_test_wh, y_pred_wh))


# Function to predict workhours
def predict_workhours(experience, designation):
    designation_encoded = label_encoder.transform([designation])[0]
    input_data = np.array([[experience, designation_encoded]])
    predicted_wh = knn_workhours.predict(input_data)
    return round(predicted_wh[0], 2)


# Function to predict salary using predicted workhours
def predict_salary(experience, designation):
    workhours = predict_workhours(experience, designation)
    designation_encoded = label_encoder.transform([designation])[0]
    input_data = np.array([[experience, designation_encoded, workhours]])
    predicted_salary = knn_salary.predict(input_data)
    return round(predicted_salary[0], 2), workhours


# function to get user input
def get_user_input():
    try:
        experience = float(input("Enter years of experience: "))
    except ValueError:
        print("Invalid input! Experience must be a number.")
        return None, None

    designation = input("Enter job designation (e.g., 'Software Engineer'): ").strip()
    valid_designations = list(label_encoder.classes_)
    match = [d for d in valid_designations if d.lower() == designation.lower()]

    if not match:
        print("Invalid Designation! Choose from:", valid_designations)
        return None, None

    return experience, match[0]


# Main Function
def main():
    experience, designation = get_user_input()
    if experience is not None and designation is not None:
        salary, workhours = predict_salary(experience, designation)
        print(f"\nPredicted Workhours for {designation} with {experience} years: {workhours} hours/week")
        print(f"Predicted Salary for {designation} with {experience} years: ${salary}")
    else:
        print("Error: Invalid input")

# Run program
if __name__ == "__main__":
    main()