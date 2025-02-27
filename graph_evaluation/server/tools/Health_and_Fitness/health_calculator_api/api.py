import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def micronutrient_requirements_get(age: int, gender: str, micronutrient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the daily requirement of a specific micronutrient using query parameters."
    age: The age of the individual for whom the micronutrient requirement is calculated. Note that this endpoint is not meant for children below 9 years.
        gender: The gender for which the micronutrient requirement is calculated. Valid values are "**male**" or "**female**"
        micronutrient: The specific micronutrient for which the requirement is calculated. Valid values include:
 "**calcium**" 
"**chromium**" 
"**copper**" 
"**fluoride**" 
"**iodine**" 
"**iron**" 
"**magnesium**" 
"**manganese**" 
"**molybdenum**" 
"**phosphorus**" 
"**selenium**" 
"**zinc**" 
"**potassium**" 
"**sodium**" or "**chloride**"
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/micronutrient"
    querystring = {'age': age, 'gender': gender, 'micronutrient': micronutrient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_bee_and_tee_get(age: int, gender: str, activity_level: str, weight: int, height: int, temperature: str='normal', stress_factor: str='none', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate Basal Energy Expenditure (BEE) and Total Energy Expenditure (TEE) using query parameters."
    age: The age of the individual in years.
        gender: The gender of the individual ('**male**' or '**female**').
        activity_level: The activity level of the individual. Choose from the following options:
'**bed-ridden**': Lying, sleeping, eating.
'**light or sedentary**': Sitting (e.g., office work), watching TV, cooking, personal care, driving a car, light walks, typical household duties.
'**moderate to active**': Standing, carrying light loads (e.g., waiting tables), longer walks, light aerobic exercises, commuting by bus.
'**heavily active**': Agricultural work, manual labor, heavy-duty cleaning, strenuous exercises performed on a regular basis.
        weight: The weight of the individual in kilograms.
        height: The height of the individual in centimeters.
        temperature: The body temperature of the individual. Choose from the following options (default is '**normal**'):
'**normal**'
'**>= 100.4°F or 38°C**'
'**>= 102.2°F or 39°C**'
'**>= 104°F or 40°C**'
'**>= 105.8°F or 41°C**'
        stress_factor: The stress factor that may affect energy expenditure. Choose from the following options (default is '**none**'):
'**none**'
'**solid tumor**'
'**leukemia/lymphoma**'
'**inflammatory bowel disease**'
'**liver disease**'
'**burns**'
'**pancreatic disease**'
'**general surgery**'
'**transplantation**'
'**sepsis**'
'**abscess**'
'**other infection**'
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bee"
    querystring = {'age': age, 'gender': gender, 'activity_level': activity_level, 'weight': weight, 'height': height, }
    if temperature:
        querystring['temperature'] = temperature
    if stress_factor:
        querystring['stress_factor'] = stress_factor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_tdee_get(activity_level: str, gender: str, height: int, age: int, weight: int, equation: str='mifflin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the Total Daily Energy Expenditure (TDEE) of an individual using query parameters."
    activity_level: The activity level of the individual. Choose from the following options:
'**sedentary**' (little to no exercise)
'**light exercise**' (light exercise or sports 1-3 days a week)
'**moderate exercise**' (moderate exercise or sports 3-5 days a week)
'**hard exercise**' (hard exercise or sports 6-7 days a week)
'**physical job**' (physically active job or training)
'**professional athlete**' (professional athlete or highly active job)
        gender: The gender of the individual ('**male**' or '**female**').
        height: The height of the individual in centimeters.
        age: The age of the individual in years.
        weight: The weight of the individual in kilograms.
        equation: The equation used to calculate BMR. Choose from the following options (default is '**mifflin**'):
'**mifflin**' (Mifflin-St Jeor Equation)
'**harris**' (Harris-Benedict Equation)
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/tdee"
    querystring = {'activity_level': activity_level, 'gender': gender, 'height': height, 'age': age, 'weight': weight, }
    if equation:
        querystring['equation'] = equation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def estimated_energy_requirement_get(height: int, age: int, activity_level: str, weight: int, gender: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates the estimated energy requirement (EER) for maintaining energy balance in healthy, normal-weight individuals."
    height: The height of the individual in centimeters.
        age: The age of the individual in years.
        activity_level: The physical activity level, which can be one of the following:
'**sedentary**' (for a sedentary lifestyle)
'**low active**' (for a low active lifestyle)
'**active**' (for an active lifestyle)
'**very active**' (for a very active lifestyle)
        weight: The weight of the individual in kilograms.
        gender: The gender of the individual ('**male**' or '**female**').
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/eer"
    querystring = {'height': height, 'age': age, 'activity_level': activity_level, 'weight': weight, 'gender': gender, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_eag_get(hba1c: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the eAG value based on the provided HbA1c percentage using a GET request."
    hba1c:  (**float**): The HbA1c value as a percentage.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/eag"
    querystring = {'hba1c': hba1c, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def body_frame_size_index_bfsi(sex: str, wrist: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the Body Frame Size Index using query parameters."
    sex: Sex of the individual ('**male**' or '**female**').
        wrist: Wrist circumference of the individual in **centimeters**.
        height: Height of the individual in **centimeters**.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bfsi"
    querystring = {'sex': sex, 'wrist': wrist, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def body_adiposity_index_bai(sex: str, age: int, hip: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates Body Adiposity Index (BAI) using query parameters."
    sex: Gender of the individual (**male** or **female**).
        age: Age of the individual.
        hip: Hip circumference in **centimeters**.
        height: Height in **meters**.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bai"
    querystring = {'sex': sex, 'age': age, 'hip': hip, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adjusted_body_weight_ajbw_imperial(sex: str, height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate AjBW and IBW based on the provided sex, height, and weight in the imperial system."
    sex: The sex of the person. Allowed values: **male**, **female**.
        height: The height of the person in **inches**.
        weight: The weight of the person in **pounds**.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/ajbw_imp"
    querystring = {'sex': sex, 'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adjusted_body_weight_ajbw(sex: str, height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate AjBW and IBW based on the provided sex, height, and weight."
    sex: The sex of the person. Allowed values: **male**, **female**.
        height: The height of the person in **centimeters**.
        weight: The weight of the person in **kilograms**.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/ajbw"
    querystring = {'sex': sex, 'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def a_body_shape_index_absi(age: int, weight: int, sex: str, height: int, waist_circumference: int, unit: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The ABSI Calculator endpoint allows you to calculate the A Body Shape Index (ABSI) and its corresponding z-score and mortality risk based on the provided parameters."
    age: The age of the individual in years.
        weight: The weight of the individual in kilograms. for '**metric**' unit, and pounds for '**imperial**' unit
        sex: The gender of the individual. Accepted values are '**male**' and '**female**'.
        height: The height of the individual in centimeters for '**metric**' unit, and inches for '**imperial**' unit.
        waist_circumference: The waist circumference of the individual in centimeters for '**metric**' unit, and inches for '**imperial**' unit.
        unit: The unit of measurement used for height, weight, and waist circumference. Accepted values are '**metric**' (default) and 'imperial'. If '**imperial**' is used, the height should be in inches, weight in pounds, and waist circumference in inches.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/absi"
    querystring = {'age': age, 'weight': weight, 'sex': sex, 'height': height, 'waist_circumference': waist_circumference, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def body_mass_index(weight: int, height: int, units: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the BMI based on the provided height and weight parameters."
    weight: The weight in **kilograms**. Required.
        height: The height in **centimeters**. Required.
        units: The desired units of measurement to implement in the JSON Response. Possible values are **metric** (default) or **imperial**. (Optional).
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bmi"
    querystring = {'weight': weight, 'height': height, }
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bmi_imperial(height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the BMI based on the provided height and weight parameters in imperial units."
    height: The height in **inches**. Required
        weight: The weight in **pounds**. Required
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bmi/imperial"
    querystring = {'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fat_free_mass_index_ffmi(body_fat: int, sex: str, unit: str, height: int, weight: int, format: str='yes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This FFMI (Fat-Free Mass Index) Calculator endpoint allows you to calculate the FFMI score, which describes the amount of muscle mass in relation to height and weight. FFMI is part of the family of body indexes, together with well-known and similar BMI. However, FFMI is more precise than BMI and provides information about somebody's condition and health."
    body_fat: The body fat percentage of the individual.
        sex: The gender of the individual (**male** or **female**).
        unit: The unit of measurement for height and weight. Possible values are \"**metric**\" (default) and \"**imperial**\".
        height: The height of the individual. For **metric** units, use height in **centimeters**, and for **imperial** units, use height in **inches**.
        weight: The weight of the individual. For **metric** units, use weight in **kilograms**, and for **imperial** units, use weight in **pounds**.
        format: The format of the response. Possible values are \"**yes**\" (default) or \"**no**\". If \"**yes**,\" the response values will be formatted as **kg/m²** or **lb/m²** based on the unit of calculation. If \"**no**,\" the response values will be provided without formatting.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/ffmi"
    querystring = {'body_fat': body_fat, 'sex': sex, 'unit': unit, 'height': height, 'weight': weight, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def macronutrient_distribution(dietary_preferences: str, body_composition_goal: str, activity_level: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the optimal distribution of macronutrients (carbohydrates, proteins, and fats) based on factors such as activity level, body composition goals, and dietary preferences."
    dietary_preferences: The dietary_preferences parameter allows users to specify their dietary preferences. It can be any string value representing the individual's dietary choices or restrictions, such as \"**vegetarian**,\" \"**vegan**,\" \"**pescatarian**,\" or \"**gluten-free**.\"
        body_composition_goal: The body_composition_goal parameter accepts the following values:

**weight_loss** - Goal of losing weight.
**maintenance** - Goal of maintaining current weight.
**muscle_gain** - Goal of gaining muscle.

        activity_level: The activity_level parameter accepts the following values:

**sedentary** - Little to no exercise.
**moderately_active** - Moderate exercise/sports 3-5 days/week.
**very_active** - Hard exercise/sports 6-7 days/week.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/mnd"
    querystring = {'dietary_preferences': dietary_preferences, 'body_composition_goal': body_composition_goal, 'activity_level': activity_level, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def target_heart_rate(age: int, fitness_level: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the target heart rate range for cardiovascular exercise based on the user's age and fitness level. It uses the Karvonen method to determine the target heart rate zone."
    age: The age of the user in years.
        fitness_level: The fitness level of the user.

The fitness_level parameter accepts the following values:
**beginner** - Beginner fitness level.
**intermediate** - Intermediate fitness level.
**advanced** - Advanced fitness level.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/thr"
    querystring = {'age': age, 'fitness_level': fitness_level, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_water_intake(weight: int, activity_level: str, climate: str, unit: str='liters', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Daily Water Intake Recommendation endpoint calculates the daily recommended water intake based on factors such as weight, activity level, and climate. It provides flexibility by allowing you to specify the unit of measurement for the water intake, either in liters or ounces."
    weight: The weight of the individual in **kilograms (kg)**.
        activity_level: The activity level of the individual. 

The activity_level parameter accepts the following values:
**sedentary** - Little to no exercise
**lightly_active** - Light exercise/sports 1-3 days/week
**moderately_active** - Moderate exercise/sports 3-5 days/week
**very_active** - Hard exercise/sports 6-7 days/week
**extra_active** - Very hard exercise/sports and physical job or 2x training
        climate: The climate in which the individual is located.

The climate parameter accepts the following values:
**normal** - Average climate
**hot** - Hot climate
**cold** - Cold climate
        unit: The unit of measurement for the water intake. 
(Default) **ounces**
 Specify **liters** to get the result in liters instead.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/dwi"
    querystring = {'weight': weight, 'activity_level': activity_level, 'climate': climate, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_caloric_needs(age: int, height: int, weight: int, goal: str, gender: str, activity_level: str, equation: str='mifflin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a simple and efficient way to calculate daily caloric needs based on various factors such as age, weight, height, activity level, and goal. It offers different formulas or approaches for caloric needs estimation, including the popular Harris-Benedict equation and Mifflin-St. Jeor equation."
    age: The age of the person in years.
        height: The height of the person in **centimeters**.
        weight: The weight of the person in **kilograms**.
        goal: The goal of the person. Valid values are \"**weight_loss**\", \"**maintenance**\", or \"**weight_gain**\".
        gender: The gender of the person. Valid values are \"**male**\" or \"**female**\".
        activity_level: The activity level of the person. Valid values are \"**sedentary**\", \"**lightly_active**\", \"**moderately_active**\", \"**very_active**\", or \"**extra_active**\".
        equation: The equation to use for caloric needs estimation. Valid values are \"**harris**\" (default) or \"**mifflin**\".
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/dcn"
    querystring = {'age': age, 'height': height, 'weight': weight, 'goal': goal, 'gender': gender, 'activity_level': activity_level, }
    if equation:
        querystring['equation'] = equation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ideal_body_weight(body_frame: str, height: int, gender: str, formula: str='hamwi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to calculate the ideal weight range based on factors like height, body frame size, and gender. The endpoint provides different formulas and approaches for ideal weight estimation, such as the Hamwi method and the Devine formula."
    body_frame: The body frame size of the person. It can be one of the following values: \"**small**\", \"**medium**\", or \"**large**\".
        height: The height in **centimeters (cm)** of the person for whom you want to calculate the ideal weight.
        gender: The gender of the person. It can be either \"**male**\" or \"**female**\".
        formula: You can include an optional query parameter to specify the formula or approach for ideal weight estimation. It can be one of the following values:
\"**hamwi**\" (default): The Hamwi method for ideal weight calculation.
\"**devine**\": The Devine formula for ideal weight calculation.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/ibw"
    querystring = {'body_frame': body_frame, 'height': height, 'gender': gender, }
    if formula:
        querystring['formula'] = formula
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basal_metabolic_rate_bmr(gender: str, age: int, height: int, weight: int, equation: str='mifflin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to calculate the Basal Metabolic Rate (BMR) based on age, weight, height, and gender parameters. The BMR represents the number of calories needed to maintain basic bodily functions at rest."
    gender: The gender (either "**male**" or "**female**").
        age: The age in **years**.
        height: The height in **centimeters**.
        weight: The weight in **kilograms**.
        equation: (optional string): The equation to use for the calculation. Valid options are "**mifflin**" (default) or "**harris**".
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bmr"
    querystring = {'gender': gender, 'age': age, 'height': height, 'weight': weight, }
    if equation:
        querystring['equation'] = equation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bodyfat(age: int, gender: str, weight: int, height: int, unit: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the body fat percentage based on the provided gender, age, height, and weight parameters."
    age: The age of the person in years. Required.
        gender: The gender of the person. Possible values are **male** or **female**. Required.
        weight: The weight in **kilograms**. Required.
        height: The height in **centimeters**. Required.
        unit: The desired units of measurement to implement in the JSON Response. Possible values are **metric** (default) or **imperial**. (Optional).
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bodyfat"
    querystring = {'age': age, 'gender': gender, 'weight': weight, 'height': height, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bodyfat_imperial(height: int, gender: str, weight: int, age: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the body fat percentage based on the provided gender, age, height, and weight parameters in imperial units."
    height: The height of the person in **inches**. Required.
        gender: The gender of the person. Must be either '**male**' or '**female**'. Required.
        weight: The weight of the person in **pounds**. Required.
        age: The age of the person in **years**. Required.
        
    """
    url = f"https://health-calculator-api.p.rapidapi.com/bodyfat/imperial"
    querystring = {'height': height, 'gender': gender, 'weight': weight, 'age': age, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "health-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

