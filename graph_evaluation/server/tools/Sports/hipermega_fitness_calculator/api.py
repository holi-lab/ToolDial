import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_calories_burnt(restingheartbeatsperminute: int, kilograms: int, age: int, treadmill: bool, slope: str, meters: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the Calories Burnt.
		
		Options:
		
		meters: Distance run in meters, not factoring in altitude.
		slope: The slope in percentage. -0.015 means that the run was downhill with a -1.5% slope.
		treadmill: Whether the run was on a treadmill. Because there is no air resistance, fewer calories will be burnt. Optional. Default value: false.
		age: The age of the runner.
		restingHeartBeatsPerMinute: The resting heart heart beats per minute rate. Normal rate is between 60-100bpm. Athletes have a lower rate and therefore burn fewer calories.
		kilograms: Weight of the runner in kilograms."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/caloriesBurnt"
    querystring = {'restingHeartBeatsPerMinute': restingheartbeatsperminute, 'kilograms': kilograms, 'age': age, 'treadmill': treadmill, 'slope': slope, 'meters': meters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate_the_fitcalc(age: str, bodyfatpercentage: int, bodytype: str, dailyactivitylevel: int, height: int, weight: str, gender: str, goal: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get perfect fitness diet macros and infos about how you can reach your goals just by passing an object.
		
		i.e:
		gender: 'male',
		    weight: 70,
		    height: 170,
		    age: 28,
		    dailyActivityLevel: 1.55,
		    bodyFatPercentage: 13,
		    bodyType: 'meso',
		    goal: 'mass-gain'
		
		**Parameters:**
		gender - male / female
		weight - (kg)
		height - (cm)
		age - (years)
		dailyActivityLevel - 1.00 (sedentary) / 1.35 (normal desk job) / 1.45 (3x per week training + normal desk job) / 1.50 (3x per week training + active work) / 1.55 (athlete & bodybuilder (5 x per week training) +normal desk job) / 1.65 (athlete & bodybuilder (5 x per week training) + active work) / 1.75 (pro athlete (5+ per week training)) / 1.85 (Endurance athlete)
		bodyFatPercentage - (integer)
		bodyType - ectomorph / meso-ecto / meso / meso-endo / endo
		goal - mass-gain / fat-loss / maintenance"
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/fitCalc"
    querystring = {'age': age, 'bodyFatPercentage': bodyfatpercentage, 'bodyType': bodytype, 'dailyActivityLevel': dailyactivitylevel, 'height': height, 'weight': weight, 'gender': gender, 'goal': goal, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_blood_alcohol_content_bac(wine: int, gender: str, weight: int, liquor: int, other: str, beer: int, timesincelastdrink: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint returns the Blood Alcohol Content (BAC).
		
		**Parameters **
		(gender ==> String, weight ==> Number, timeSinceLastDrink ==> Number, consumptionData ==> Object).
		consumptionData is required in the following format ==> Object
		
		**Note:** consumptionData = wine,liquor,other - > Pass it as a String for each one of them.
		
		All parameters required.
		Returns BAC (%) ==> Number.
		Gender is not case sensitive."
    other: Please separate them by comma.
        
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/bac"
    querystring = {'wine': wine, 'gender': gender, 'weight': weight, 'liquor': liquor, 'other': other, 'beer': beer, 'timeSinceLastDrink': timesincelastdrink, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_total_daily_energy_expenditure_tdee(weight: int, activity: str, age: int, gender: str, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns TOTAL DAILY ENERGY EXPENDITURE (TDEE).
		
		**Parameters **
		(gender ==> String, age ==> Number, height ==> Number, weight ==> Number, activity ==> String).
		activity is one of the following [sedentary, light, moderate, active, extreme].
		All parameters required.
		Returns TDEE for balanced goal ==> Number.
		Gender, activity are not case sensitive."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/tdee"
    querystring = {'weight': weight, 'activity': activity, 'age': age, 'gender': gender, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_macros(goal: str, age: int, activity: str, height: int, gender: str, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "THIS ENDPOINT RETURNS THE MACROS .
		
		**Parameters **
		(gender ==> String, age ==> Number, height ==> Number, weight ==> Number, activity ==> String, goal ==> String).
		Activity is one of the following [sedentary, light, moderate, active, extreme].
		Goal is one of the following [balance, mildWeightLoss, mildWeightGain, heavyWeightLoss, heavyWeightGain]
		All parameters required."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/macros"
    querystring = {'goal': goal, 'age': age, 'activity': activity, 'height': height, 'gender': gender, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_ideal_body_weight_ibw(gender: str='male', weight: int=70, height: int=170, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns IDEAL BODY WEIGHT (IBW).
		
		**Parameters **
		(gender ==> String, height ==> Number).
		All parameters required
		Returns idealBodyWeight ==> Number.
		Gender is not case sensitive."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/ibw"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    if weight:
        querystring['weight'] = weight
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_body_fat_percentage_bfp(neck: int, weight: int, hip: int, height: int, gender: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Returns a BODY FAT PERCENTAGE (BFP).
		
		**Parameters **
		(gender ==> String, height ==> Number, weight ==> Number, neck ==> Number, waist ==> Number, hip ==> Number).
		All parameters required (exception for hip measurement).
		Returns BFP ==> Number. (percentage value)
		Gender is not case sensitive."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/bfp"
    querystring = {'neck': neck, 'weight': weight, 'hip': hip, 'height': height, 'gender': gender, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_calories_needed(activity: str, height: int, weight: int, age: int, gender: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns calories Needs for several cases of goals from [balance, mildWeightLoss, mildWeightGain, heavyWeightLoss, heavyWeightGain] ==> Object.
		
		**Parameters **
		(gender ==> String, age ==> Number, height ==> Number, weight ==> Number, activity ==> String).
		activity is one of the following [sedentary, light, moderate, active, extreme].
		All parameters required."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/caloriesneeds"
    querystring = {'activity': activity, 'height': height, 'weight': weight, 'age': age, 'gender': gender, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_basal_metabolic_rate_bmr(weight: int, height: int, age: int, gender: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns Basal Metabolic.
		All measurements are in metric unit (cm, kg) as of now.
		
		**Parameters **
		(gender ==> String, age ==> Number, height ==> Number, weight ==> Number).
		All parameters required.
		Returns BMR ==> Number.
		Gender is not case sensitive."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/bmr"
    querystring = {'weight': weight, 'height': height, 'age': age, 'gender': gender, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_body_mass_index_bmi(weight: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint returns the Body Mass Index.
		Parameters (height ==> Number, weight ==> Number).
		All parameters required.
		Returns BMI ==> Number.
		Gender is not case sensitive."
    
    """
    url = f"https://hipermega-fitness-calculator.p.rapidapi.com/bmi"
    querystring = {'weight': weight, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hipermega-fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

