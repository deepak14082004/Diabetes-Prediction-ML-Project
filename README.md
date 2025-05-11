Diabetes Prediction Using Machine Learning

This project uses a machine learning model to predict whether a patient is diabetic based on medical diagnostic measurements. It was developed as part of an internship with InternPE.

Dataset

The dataset includes the following features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age


The target column is:

Outcome:

1 → Diabetic

0 → Non-Diabetic



Tools & Technologies

Python

Pandas

scikit-learn

Logistic Regression


How It Works

1. The model is trained on the diabetes.csv dataset.


2. It takes new patient data from a separate CSV file (new_patients.csv).


3. It predicts whether each patient is diabetic or not.


4. The result is printed and saved in a new file: patients_with_predictions.csv.



How to Use

1. Clone the repository:

git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction


2. Install required libraries:

pip install pandas scikit-learn


3. Place your patient data in diabetes.csv using this format:

Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
13,145,82,19,110,22.2,0.245,57

anything data can replace you in the diabtese.csv file and check outcome if 1 then output will show diabetic otherwise non diabetic


4. Run the script:

python predict.py



Output

The script will add a Prediction column to your CSV:

Diabetic

Not Diabetic


And save it as patients_with_predictions.csv.

Acknowledgements

This project was developed during my internship at InternPE.
Special thanks to the InternPE team for the opportunity and guidance!

License

This project is open-source and available under the MIT License.
