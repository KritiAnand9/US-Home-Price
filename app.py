from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__)
df = pd.read_excel("output_US_Home_Prices.xlsx")
model = sm.tsa.ARIMA(df['SPI'], order=(1, 1, 1))
result = model.fit()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict_spi', methods=['POST'])
def predict_spi():
    try:
        selected_month = request.form['selected_month']
        selected_month = pd.to_datetime(selected_month, format='%Y-%m-%d')
        forecast = result.get_forecast(steps=1, exog=[selected_month])
        forecast_value = forecast.predicted_mean.iloc[0]
        response = {
            'selected_month': selected_month.strftime('%Y-%m-%d'),
            'forecast_value': forecast_value
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)