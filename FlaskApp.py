from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField,SubmitField,validators
from wtforms.validators import DataRequired
import os
import pickle
import numpy as np 

app = Flask(__name__)
app.config['SECRET_KEY']='kah'

try:
    with open('WatchPrice.pkl', 'rb') as file:
        watch_model = pickle.load(file)
    print("Watch price model loaded successfully.")
    with open('lable.pkl', 'rb') as file:
        lbbrn= pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure 'watch_price_model.pkl' exists.")
except Exception as e:
    print(f"Error loading model: {e}")

class WatchPredictionForm(FlaskForm):
    brand = SelectField('Brand',
                    validators=[DataRequired()],
                    choices=[
                        ('Rolex', 'Rolex'), ('Omega', 'Omega'), ('Tag Heuer', 'Tag Heuer'),
                        ('Breitling', 'Breitling'), ('Cartier', 'Cartier'), ('Jaeger-LeCoultre', 'Jaeger-LeCoultre'),
                        ('Seiko', 'Seiko'), ('Citizen', 'Citizen'), ('Tissot', 'Tissot'), ('Hamilton', 'Hamilton'),
                        ('Longines', 'Longines'), ('Oris', 'Oris'), ('Bell & Ross', 'Bell & Ross'), ('Breguet', 'Breguet'),
                        ('Audemars Piguet', 'Audemars Piguet'), ('Vacheron Constantin', 'Vacheron Constantin'),
                        ('Panerai', 'Panerai'), ('Tudor', 'Tudor'), ('IWC', 'IWC'), ('Montblanc', 'Montblanc'),
                        ('Blancpain', 'Blancpain'), ('Zenith', 'Zenith'), ('Piaget', 'Piaget'),
                        ('Ulysse Nardin', 'Ulysse Nardin'), ('Jaquet Droz', 'Jaquet Droz'), ('Bulgari', 'Bulgari'),
                        ('Chopard', 'Chopard'), ('Girard-Perregaux', 'Girard-Perregaux'),
                        ('Glashütte Original', 'Glashütte Original'), ('Hublot', 'Hublot'),
                        ('Patek Philippe', 'Patek Philippe'), ('Bulova', 'Bulova'), ('Sinn', 'Sinn'),
                        ('A. Lange & Sohne', 'A. Lange & Sohne'), ('Rado', 'Rado'),
                        ('Frederique Constant', 'Frederique Constant'), ('TAG Heuer', 'TAG Heuer'),
                        ('Baume & Mercier', 'Baume & Mercier'), ('A. Lange & Söhne', 'A. Lange & Söhne')
                    ],
                    render_kw={"placeholder": "e.g., Rolex, Omega, Seiko, Titan, Casio"})
    water_resistance=StringField('Water Resistance in meters:',validators=[validators .DataRequired()])
    power_reserve=StringField('Power Reserve in hours:',validators=[validators .DataRequired()])
    submit = SubmitField('Predict Price')

def preprocess_input(brand, water_resistance_str,power_reserve_str):
    try:
        water_resistance_value = int(water_resistance_str)
        power_reserve_value = int(power_reserve_str)
    except ValueError:
        water_resistance_value = 0
        power_reserve_value = 0
    brand_feature=lbbrn.fit_transform([brand])
    return np.array([[brand_feature[0], water_resistance_value,power_reserve_value]])

@app.route('/', methods=['GET', 'POST'])
def index():
    form = WatchPredictionForm()
    prediction_result = None
    error_message = None

    if form.validate_on_submit():
        if watch_model is None:
            error_message = "Machine learning model not loaded. Please check server logs for errors."
        else:
            try:
                brand = form.brand.data
                water_resistance = form.water_resistance.data
                power_reserve = form.power_reserve.data
                input_for_model = preprocess_input(brand, water_resistance,power_reserve)
                predicted_price = watch_model.predict(input_for_model)
                prediction_result = {
                    'brand': brand,
                    'water_resistance': form.water_resistance, 
                    'power_reserve': form.power_reserve,
                    'price': predicted_price
                }
                
            except Exception as e:
                error_message = f"An error occurred during prediction: {str(e)}. Please check your input."
                print(f"Prediction error: {e}")

    return render_template('model.html', form=form, prediction_result=prediction_result, error_message=error_message)
app.run()
