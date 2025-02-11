#import libraries
from flask import Flask, request, render_template,jsonify # Import flask libraries for API
import numpy as np # To deal with numbers
import pickle # To deal with model
import joblib # To deal with model
import random
import pandas as pd # To deal wuth data
from datetime import datetime # To deal with date time. etc

#Initialize the flask App
app = Flask(__name__)


reference_data = pd.read_csv('datasets/clean_category_data.csv')



#default page of our web-app
@app.route('/')
def landing():
    return render_template('1.landing.html')


# Signup page
@app.route('/signup',methods=['POST'])
def signup():
    if request.method == 'POST':
        return render_template('2.signup.html')

# Signup success page
@app.route('/signupsuccess',methods=['POST'])
def signupsuccess():
    if request.method == 'POST':
        credentials = [(x) for x in request.form.values()]
        print(credentials)
        username = credentials[0]
        password = credentials[1]
        print(type(username))

        file = open("Username/username.txt", "w")
        a = file.write(username)
        file.close()

        file = open("Password/password.txt", "w")
        a = file.write(password)
        file.close()
        return render_template('3.signupsuccess.html')



# login page
@app.route('/login',methods=['POST'])
def login():
    if request.method == 'POST':
        return render_template('4.login.html')

# backup login page
@app.route('/login2')
def login2():
    return render_template('4.login.html')

# dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('14.dashboard.html')

# precautions page
@app.route('/preco')
def preco():
    return render_template('15.preco.html')


# main home page
@app.route('/home',methods=['POST'])
def home():
    if request.method == 'POST':
        lcredentials = [(x) for x in request.form.values()]
        print(lcredentials)
        lusername = lcredentials[0]
        lpassword = lcredentials[1]
        print(type(lusername))

        f = open("Username/username.txt", "r")
        username = f.read()
        f = open("Password/password.txt", "r")
        password = f.read()
        print(lusername, username, lpassword, password)
        cities = reference_data['City'].unique().tolist()

        if username==lusername and password==lpassword:
            print('match')
            template = '6.home.html'
        elif username!=lusername or password!=lpassword:
            print('No')
            template = '5.loginfailed.html'

        return render_template(template)

# index page
@app.route('/check')
def check():
    cities = reference_data['City'].unique().tolist()
    return render_template('7.index.html', cities=cities)


@app.route('/check2')
def check2():
    cities = reference_data['City'].unique().tolist()
    return render_template('7.index2.html', cities=cities)

# about page
@app.route('/about')
def about():
    return render_template('13.about.html')

# backup home page
@app.route('/home2')
def home2():
    return render_template('6.home.html')

# result page routing
@app.route('/predict', methods=['POST'])
def predict_air_quality():
    city = request.form['city']
    date = request.form['date']
    obj = datetime.strptime(date, '%Y-%m-%d')
    if obj.year>2018:
      dif = obj.year - 2018
      nyear = obj.year - dif
      nyear = 2020
      nobj = obj.replace(year=nyear)
      ndate = nobj.strftime('%Y-%m-%d')
    else:
      ndate = date


    model_type = request.form['model']  # Get selected model type
    print('Hehe', city)
    print(ndate)


    # Fetch data for selected city and date
    data = reference_data[(reference_data['City'] == city) & (reference_data['Date'] == ndate)]
    print(data)

    # Extract features
    features = data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']]
    print('Features are', features)
    O3 = features['O3'].iloc[0]
    NOx = features['NOx'].iloc[0]
    CO = features['CO'].iloc[0]
    print('GW: ', O3, NOx, CO)

    if O3>75 or NOx>75 or CO>15:
        level = 'HIGH'
        cause =  'This will add to Global Warming'
    else:
        level = 'LOW'
        cause = 'Does not cause Global Warming'


    # Get pollutant concentrations
    pollutants = list(features.columns)
    concentrations = features.values.tolist()
    flattened_concentrations = [val for sublist in concentrations for val in sublist]
    pollutant_concentrations = dict(zip(pollutants, flattened_concentrations))


    if model_type == 'linear':
        model = joblib.load('models/linear_reg_model.pkl')
    elif model_type == 'random_forest':
        model = joblib.load('models/random_forest_reg.pkl')
    elif model_type == 'XGBoost':
        model = joblib.load('models/xgboost_reg.pkl')
    elif model_type == 'lightBGM':
        model = joblib.load('models/lgb_reg.pkl')
    else:
        return "Invalid model selection."
    # Predict AQI
    print('Predicting.........')
    print(features)
    result = model.predict(features)
    print(result)
    predict_air_quality_index = round(result[0], 2)
    print('Predicted')

    # Map AQI to category
    category_mapping = {
        (0, 50): "Good",
        (51, 100): "Satisfactory",
        (101, 200): "Moderate",
        (201, 300): "Poor",
        (301, 400): "Very Poor",
        (401, float('inf')): "Severe"
    }

    # Iterate through the category_mapping dictionary
    predicted_category = None
    for range_, category in category_mapping.items():
        if predict_air_quality_index <= range_[1]:
            predicted_category = category
            break

    # If predicted_category is still None, the value is out of range
    if predicted_category is None:
        predicted_category = "Undefined"

    # Fetch health implications for categories 3 to 6 of each pollutant
    health_implications = {
        'PM2.5': {
            3: "Poor - Aggravation of existing respiratory conditions, leading to more frequent symptoms and decreased lung function. Increased risk of respiratory infections such as pneumonia and bronchitis",
            4: "Very Poor - Greater likelihood of experiencing severe respiratory symptoms, including difficulty breathing, chest tightness, and coughing. Worsening of cardiovascular conditions, potentially leading to heart attacks or strokes, particularly in vulnerable populations such as the elderly and those with pre-existing heart disease.",
            5: "Severe - Acute respiratory distress, with symptoms becoming more pronounced and widespread in the population. Increased risk of hospital admissions due to respiratory and cardiovascular issues. Exacerbation of chronic health conditions, posing a significant threat to public health.",
            6: "Hazardous - Critical respiratory and cardiovascular effects, including acute respiratory distress syndrome (ARDS) and cardiac events. Immediate and widespread health impacts, with potentially life-threatening consequences. Emergency measures may be required to protect public health, such as evacuation or sheltering in place."
        },
        'PM10': {
            3: "Poor- Increased risk of respiratory issues, including coughing, throat irritation, and shortness of breath, especially during physical activity or prolonged exposure.",
            4: "Very Poor - Aggravation of existing respiratory conditions, leading to more frequent symptoms and reduced lung function. Greater susceptibility to respiratory infections such as bronchitis and pneumonia.",
            5: "Severe - High respiratory symptoms in vulnerable individuals, including exacerbation of asthma and COPD. Increased risk of respiratory infections and other health issues, requiring medical attention for some individuals",
            6: "Hazardous - Severe respiratory distress, with symptoms becoming widespread and potentially life-threatening. Higher likelihood of hospital admissions due to respiratory issues and exacerbation of chronic health conditions. Emergency measures may be necessary to protect public health, including limiting outdoor activities and implementing air quality alerts."
        },
        'CO': {
            3: "Poor - Headaches, dizziness, and nausea can be experienced by sensitive individuals. Continued exposure may lead to more severe symptoms.",
            4: "Very Poor - Headaches, dizziness, and nausea become more common. People with heart disease may experience chest pain and reduced exercise tolerance.",
            5: "Severe - Extreme headaches, dizziness, and nausea are expected. Individuals with heart disease may experience significant aggravation of symptoms, including chest pain and potential cardiac arrhythmias.",
            6: "Hazardous - Life-threatening symptoms such as loss of consciousness, seizures, and coma may occur rapidly. Immediate medical attention is required, and exposure to CO at this level can be fatal without prompt treatment."
        },
        'NO2': {
            3: "Poor - Increased respiratory symptoms may occur in sensitive individuals, including worsening of asthma and other respiratory conditions. Long-term exposure may lead to decreased lung function in susceptible populations.",
            4: "Very Poor - Increased respiratory symptoms such as coughing, wheezing, and shortness of breath may occur in the general population. Individuals with pre-existing respiratory or cardiovascular conditions may experience exacerbation of symptoms.",
            5: "Severe - Significant respiratory symptoms, including severe coughing, wheezing, and shortness of breath, are expected. Risk of respiratory infections may increase due to reduced lung function and compromised immune response.",
            6: "Hazardous - Individuals may experience chest pain, palpitations, and other cardiovascular symptoms. Long-term exposure to levels this high can have serious health consequences, including permanent lung damage and increased risk of cardiovascular events. Immediate medical attention is required."
        },
        'SO2': {
            3: "Poor - Increased respiratory symptoms may occur, particularly in sensitive individuals, including worsening of asthma, increased frequency of coughing, chest tightness, and difficulty breathing.",
            4: "Very Poor - Respiratory symptoms may become more pronounced, affecting both sensitive and healthy individuals, leading to severe coughing, wheezing, shortness of breath, and chest discomfort.",
            5: "Severe - Individuals may experience significant respiratory distress, with symptoms such as severe coughing fits, difficulty breathing even at rest, wheezing audible without a stethoscope, and tightness in the chest.",
            6: "Hazardous - Severe respiratory symptoms may occur, leading to life-threatening conditions such as acute respiratory distress syndrome (ARDS), respiratory failure, severe bronchospasm, suffocation, and death, especially in vulnerable populations such as children, the elderly, and individuals with pre-existing respiratory conditions."
        },
        'O3': {
            3: "Moderate - Respiratory symptoms may worsen, particularly in sensitive individuals. This can include increased asthma symptoms, more frequent coughing, chest tightness, and slight difficulty in breathing.",
            4: "Poor - Both sensitive and healthy individuals may experience pronounced respiratory symptoms. These can include severe coughing, wheezing, moderate difficulty breathing, and discomfort in the chest.",
            5: "Very Poor - Significant respiratory distress may occur, characterized by severe coughing fits, difficulty breathing even at rest, audible wheezing, and noticeable chest tightness.",
            6: "Severe - Severe respiratory symptoms pose life-threatening risks, especially to vulnerable populations like children, the elderly, and those with existing respiratory conditions. Conditions such as acute respiratory distress syndrome (ARDS), respiratory failure, and suffocation may occur."
        }
    }

    # Pass health implications to the result.html template
    pollutant_health_implications = {}
    for pollutant, categories in health_implications.items():
        pollutant_category = data.get(f"{pollutant}_category")
        if pollutant_category is not None:
            pollutant_health_implications[pollutant] = categories.get(pollutant_category.iloc[0], "")  # Get health implications if category is 3 to 6




    return render_template('8.result.html', level=level, cause=cause, city=city, date=str(date), data=data, pollutant_concentrations=pollutant_concentrations, AQI=predict_air_quality_index, category=predicted_category, health_implications=pollutant_health_implications)

@app.route('/predict2', methods=['POST'])
def predict_air_quality2():
    city = request.form['city']
    date = request.form['date']
    model_type = request.form['model']  # Get selected model type
    pm25 = float(request.form['PM2.5'])
    pm10 = float(request.form['PM10'])
    no = float(request.form['NO'])
    no2 = float(request.form['NO2'])
    nox = float(request.form['NOx'])
    nh3 = float(request.form['NH3'])
    co = float(request.form['CO'])
    so2 = float(request.form['SO2'])
    o3 = float(request.form['O3'])

    print(pm25)
    poll = [pm25, pm10, no, no2, nox, nh3, co, so2, o3]
    O3 = o3
    NOx = nox
    CO = co

    if O3>60 or NOx>30 or CO>15:
        level = 'HIGH'
        cause =  'This will add to Golbal Warming'
    else:
        level = 'LOW'
        cause = 'Does not cause Global Warming'


    # Define the expected feature names
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    # Create a dictionary for pollutant concentrations (optional, for reference
    pollutant_concentrations = dict(zip(pollutants, poll))
    # Load the trained model
    model = joblib.load('models/xgboost_reg.pkl')


    # Reshape the input to match the model's expected format
    # Assuming the model expects a 2D array of shape (n_samples, n_features)
    features = np.array(poll).reshape(1, -1)


    # Predict using the model
    result = model.predict(features)





    print(city, date)
    predict_air_quality_index = round(result[0], 2)
    print('Predicted')

    # Map AQI to category
    category_mapping = {
        (0, 50): "Good",
        (51, 100): "Satisfactory",
        (101, 200): "Moderate",
        (201, 300): "Poor",
        (301, 400): "Very Poor",
        (401, float('inf')): "Severe"
    }

    # Iterate through the category_mapping dictionary
    predicted_category = None
    for range_, category in category_mapping.items():
        if predict_air_quality_index <= range_[1]:
            predicted_category = category
            break

    # If predicted_category is still None, the value is out of range
    if predicted_category is None:
        predicted_category = "Undefined"

    # Fetch health implications for categories 3 to 6 of each pollutant
    health_implications = {
        'PM2.5': {
            3: "Poor - Aggravation of existing respiratory conditions, leading to more frequent symptoms and decreased lung function. Increased risk of respiratory infections such as pneumonia and bronchitis",
            4: "Very Poor - Greater likelihood of experiencing severe respiratory symptoms, including difficulty breathing, chest tightness, and coughing. Worsening of cardiovascular conditions, potentially leading to heart attacks or strokes, particularly in vulnerable populations such as the elderly and those with pre-existing heart disease.",
            5: "Severe - Acute respiratory distress, with symptoms becoming more pronounced and widespread in the population. Increased risk of hospital admissions due to respiratory and cardiovascular issues. Exacerbation of chronic health conditions, posing a significant threat to public health.",
            6: "Hazardous - Critical respiratory and cardiovascular effects, including acute respiratory distress syndrome (ARDS) and cardiac events. Immediate and widespread health impacts, with potentially life-threatening consequences. Emergency measures may be required to protect public health, such as evacuation or sheltering in place."
        },
        'PM10': {
            3: "Poor- Increased risk of respiratory issues, including coughing, throat irritation, and shortness of breath, especially during physical activity or prolonged exposure.",
            4: "Very Poor - Aggravation of existing respiratory conditions, leading to more frequent symptoms and reduced lung function. Greater susceptibility to respiratory infections such as bronchitis and pneumonia.",
            5: "Severe - High respiratory symptoms in vulnerable individuals, including exacerbation of asthma and COPD. Increased risk of respiratory infections and other health issues, requiring medical attention for some individuals",
            6: "Hazardous - Severe respiratory distress, with symptoms becoming widespread and potentially life-threatening. Higher likelihood of hospital admissions due to respiratory issues and exacerbation of chronic health conditions. Emergency measures may be necessary to protect public health, including limiting outdoor activities and implementing air quality alerts."
        },
        'CO': {
            3: "Poor - Headaches, dizziness, and nausea can be experienced by sensitive individuals. Continued exposure may lead to more severe symptoms.",
            4: "Very Poor - Headaches, dizziness, and nausea become more common. People with heart disease may experience chest pain and reduced exercise tolerance.",
            5: "Severe - Extreme headaches, dizziness, and nausea are expected. Individuals with heart disease may experience significant aggravation of symptoms, including chest pain and potential cardiac arrhythmias.",
            6: "Hazardous - Life-threatening symptoms such as loss of consciousness, seizures, and coma may occur rapidly. Immediate medical attention is required, and exposure to CO at this level can be fatal without prompt treatment."
        },
        'NO2': {
            3: "Poor - Increased respiratory symptoms may occur in sensitive individuals, including worsening of asthma and other respiratory conditions. Long-term exposure may lead to decreased lung function in susceptible populations.",
            4: "Very Poor - Increased respiratory symptoms such as coughing, wheezing, and shortness of breath may occur in the general population. Individuals with pre-existing respiratory or cardiovascular conditions may experience exacerbation of symptoms.",
            5: "Severe - Significant respiratory symptoms, including severe coughing, wheezing, and shortness of breath, are expected. Risk of respiratory infections may increase due to reduced lung function and compromised immune response.",
            6: "Hazardous - Individuals may experience chest pain, palpitations, and other cardiovascular symptoms. Long-term exposure to levels this high can have serious health consequences, including permanent lung damage and increased risk of cardiovascular events. Immediate medical attention is required."
        },
        'SO2': {
            3: "Poor - Increased respiratory symptoms may occur, particularly in sensitive individuals, including worsening of asthma, increased frequency of coughing, chest tightness, and difficulty breathing.",
            4: "Very Poor - Respiratory symptoms may become more pronounced, affecting both sensitive and healthy individuals, leading to severe coughing, wheezing, shortness of breath, and chest discomfort.",
            5: "Severe - Individuals may experience significant respiratory distress, with symptoms such as severe coughing fits, difficulty breathing even at rest, wheezing audible without a stethoscope, and tightness in the chest.",
            6: "Hazardous - Severe respiratory symptoms may occur, leading to life-threatening conditions such as acute respiratory distress syndrome (ARDS), respiratory failure, severe bronchospasm, suffocation, and death, especially in vulnerable populations such as children, the elderly, and individuals with pre-existing respiratory conditions."
        },
        'O3': {
            3: "Moderate - Respiratory symptoms may worsen, particularly in sensitive individuals. This can include increased asthma symptoms, more frequent coughing, chest tightness, and slight difficulty in breathing.",
            4: "Poor - Both sensitive and healthy individuals may experience pronounced respiratory symptoms. These can include severe coughing, wheezing, moderate difficulty breathing, and discomfort in the chest.",
            5: "Very Poor - Significant respiratory distress may occur, characterized by severe coughing fits, difficulty breathing even at rest, audible wheezing, and noticeable chest tightness.",
            6: "Severe - Severe respiratory symptoms pose life-threatening risks, especially to vulnerable populations like children, the elderly, and those with existing respiratory conditions. Conditions such as acute respiratory distress syndrome (ARDS), respiratory failure, and suffocation may occur."
        }
    }




    return render_template('8.result2.html', level=level, cause=cause, city=city, date=date, pollutant_concentrations=pollutant_concentrations, AQI=predict_air_quality_index, category=predicted_category)




if __name__ == "__main__":
    app.run(debug=True)
