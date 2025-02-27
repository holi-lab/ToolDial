import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def daily_calory_requirements(activitylevel: str, weight: int, gender: str, height: int, age: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Knowing your daily calorie requirements is important to achieve your final goal. You can calculate your daily calorie req. for 7 different goals."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/dailycalorie"
    querystring = {'activitylevel': activitylevel, 'weight': weight, 'gender': gender, 'height': height, 'age': age, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def burned_calorie_from_activity(weight: int, activityid: str, activitymin: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the burned calorie for a specific activity."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/burnedcalorie"
    querystring = {'weight': weight, 'activityid': activityid, 'activitymin': activitymin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def activities(intensitylevel: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the met values for activities."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/activities"
    querystring = {'intensitylevel': intensitylevel, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food_info(foodid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the food info for a food id from the food database."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/food"
    querystring = {'foodid': foodid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food_names_and_ids(subtablename: str='Fo1_2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find food table ids for a sub-table in the food database."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/foodids"
    querystring = {}
    if subtablename:
        querystring['subtablename'] = subtablename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sub_table_names(tablename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find sub-table names of a table name in the food database."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/foodids/subtablenames"
    querystring = {'tablename': tablename, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tablenames(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the more than 70 main table names of the food database."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/foodids/tablenames"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def macros_amounts(age: int, gender: str, weight: int, goal: str, height: int, activitylevel: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the amount of macro nutrients in four different categories which are balanced, low-fat, low-carbs and high-protein for a specific calorie burned. There are 6 inputs which are age, gender, height, weight, activity level and goal. 
		
		For activity level :
		    
		-     "1" : "BMR",
		-     "2" : "Sedentary: little or no exercise",
		-     "3" : "Exercise 1-3 times/week",
		-     "4" : "Exercise 4-5 times/week",
		-     "5" : "Daily exercise or intense exercise 3-4 times/week",
		-     "6" : "Intense exercise 6-7 times/week",
		-     "7" : "Very intense exercise daily, or physical job" 
		
		For goals : 
		
		-      "maintain" : "maintain weight",
		-     "mildlose" : "Mild weight loss",
		-     "weightlose" : "Weight loss",
		-     "extremelose" : "Extreme weight loss",
		-     "mildgain" : "Mild weight gain",
		-     "weightgain" : "Weight gain",
		-     "extremegain" : "Extreme weight gain""
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/macrocalculator"
    querystring = {'age': age, 'gender': gender, 'weight': weight, 'goal': goal, 'height': height, 'activitylevel': activitylevel, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def body_fat_percentage(height: int, waist: int, neck: int, gender: str, hip: int, age: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Knowing body fat percentage would be helpful to decide on a well-programmed fitness plan. You can use this endpoint with 7 parameters."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/bodyfat"
    querystring = {'height': height, 'waist': waist, 'neck': neck, 'gender': gender, 'hip': hip, 'age': age, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ideal_weight(gender: str, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find four different ideal weight scores according to four different well-known equations which are Hamwi, Devine, Miller, Robinson.  There are two parameters which are age and height(cm) values."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/idealweight"
    querystring = {'gender': gender, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bmi(age: int, height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find body mass index value (BMI) with this endpoint. You just need to enter three parameters which are age, weight (kg), and height(cm) information."
    
    """
    url = f"https://fitness-calculator.p.rapidapi.com/bmi"
    querystring = {'age': age, 'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fitness-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

