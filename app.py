# Import necessary libraries
# Import necessary libraries
# Import necessary libraries
# Import necessary libraries
# Import necessary libraries
# Import necessary libraries
# Import necessary libraries
import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

# Initialize Flask application
app = Flask(__name__, static_url_path='/static')
# Load your trained SVM model (replace 'your_model.pkl' with the actual path)
model = joblib.load('classifier.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        gender = request.form['gender']
        married = request.form['married']
        dependents = int(request.form['dependents'])
        
        # Encode 'Gender' using LabelEncoder
        gender_encoder = LabelEncoder()
        gender_encoded = gender_encoder.fit_transform([gender])

        # Encode 'Married' using LabelEncoder
        married_encoder = LabelEncoder()
        married_encoded = married_encoder.fit_transform([married])

        education = request.form['education']

        # Encode 'Education' using one-hot encoding
        education_encoded = [1 if education == 'Graduate' else 0]

        self_employed = request.form['self_employed']

        # Encode 'Self Employed' using LabelEncoder
        self_employed_encoder = LabelEncoder()
        self_employed_encoded = self_employed_encoder.fit_transform([self_employed])

        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])
        property_area = request.form['property_area']

        # Encode 'Property Area' using LabelEncoder
        property_area_encoder = LabelEncoder()
        property_area_encoded = property_area_encoder.fit_transform([property_area])

        # Make predictions using the SVM model
        prediction = model.predict([[
            dependents, applicant_income, coapplicant_income,
            loan_amount, loan_amount_term, credit_history,
            *gender_encoded,  # Include the encoded 'Gender'
            *married_encoded,  # Include the encoded 'Married'
            *self_employed_encoded,  # Include the encoded 'Self Employed'
            *education_encoded,  # Include the encoded 'Education'
            *property_area_encoded,  # Include the encoded 'Property Area'
        ]])[0]

        # Map the prediction to "Approved" or "Denied"
        prediction_label = "Approved" if prediction == 1 else "Denied"

        # Return the prediction as a rendered HTML template
        return render_template('result.html', prediction=prediction_label)
    except Exception as e:
        # Log the error for debugging purposes
        app.logger.error(str(e))
        # Return an error response with a detailed message
        return render_template('result.html', prediction='Error occurred during prediction. ' + str(e))

if __name__ == '__main__':
    app.run(debug=True)




