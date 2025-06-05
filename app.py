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
        feature_names = [
            'volatile acidity', 'residual sugar',
            'chlorides', 'total sulfur dioxide', 
            'density', 'sulphates', 'alcohol'
        ]

        features = []
        for fname in feature_names:
            form_key = fname.replace(' ', '_')
            val = request.form.get(form_key)
            if val is None or val == '':
                raise ValueError(f'Missing value for {fname}')
            features.append(float(val))

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
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(debug=True, host='0.0.0.0', port=port)
