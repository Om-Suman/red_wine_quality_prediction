from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

model = joblib.load('xgb_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define expected features in the correct order matching model training
        feature_names = [
            'volatile acidity', 'residual sugar',
            'chlorides', 'total sulfur dioxide', 
            'density', 'sulphates', 'alcohol'
        ]

        # Extract features from form, convert to float safely
        features = []
        for fname in feature_names:
            # form keys have underscores instead of spaces, so map accordingly:
            form_key = fname.replace(' ', '_')
            val = request.form.get(form_key)
            if val is None or val == '':
                raise ValueError(f'Missing value for {fname}')
            features.append(float(val))

        # Create a DataFrame to feed into the model
        sample_df = pd.DataFrame([features], columns=feature_names)

        # Get prediction (assumes model outputs 1 or 0 for Excellent/Poor)
        prediction = model.predict(sample_df)

        label = 'Excellent' if prediction[0] == 1 else 'Poor'

        return jsonify({
            'prediction_text': f'Predicted Wine Quality: {label}',
            'error': False
        })

    except Exception as e:
        return jsonify({
            'prediction_text': f'Error: {str(e)}',
            'error': True
        })


if __name__ == '__main__':
    app.run(debug=False)
