import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_body_mass_index_bmi(height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Health Status Available : **
		- Under Weight
		- Normal Weight
		- Over Weight
		- Obesity class I
		- Obesity class II
		- Obesity class III"
    
    """
    url = f"https://mega-fitness-calculator1.p.rapidapi.com/bmi"
    querystring = {'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-fitness-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_total_daily_energy_expenditure_tdee(weight: int, gender: str, age: int, activitylevel: str, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TDEE is the total amount of calories burned when the activity rate is taken into account..
		
		**Activity Level Possible Values**
		-  se => sedentary
		- la => lightly active
		- ma => moderately active
		- va => very active
		- ea => extra active"
    
    """
    url = f"https://mega-fitness-calculator1.p.rapidapi.com/tdee"
    querystring = {'weight': weight, 'gender': gender, 'age': age, 'activitylevel': activitylevel, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-fitness-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_body_shape_index_absi(gender: str, height: int, age: int, waist: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ABSI index is used to estimate the risk of premature mortality based on health factors.
		**Available Mortality Risk Status**
		- Very Low
		- Low 
		- Average
		- High
		- Very High"
    
    """
    url = f"https://mega-fitness-calculator1.p.rapidapi.com/absi"
    querystring = {'gender': gender, 'height': height, 'age': age, 'waist': waist, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-fitness-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_waist_hip_ratio_whr(waist: int, gender: str, hip: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "WHR is used to determine fat distribution in body and indicate the person’s overall health.
		**Available Shapes Status**
		- Pear
		- Apple
		*Available risk level Status*
		- Low Health Risk
		- Moderate Health Risk
		- High Health Risk"
    
    """
    url = f"https://mega-fitness-calculator1.p.rapidapi.com/whr"
    querystring = {'waist': waist, 'gender': gender, 'hip': hip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-fitness-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ideal_body_weight_ibw(gender: str, height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "*Get IBW in 4 different Formulas : *
		- Robinson
		-  Miller
		- Devine
		- Hamwi"
    
    """
    url = f"https://mega-fitness-calculator1.p.rapidapi.com/ibw"
    querystring = {'gender': gender, 'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-fitness-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_body_fat_percentage_bfp(weight: int, age: int, height: int, gender: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find BFP value , Total Fat Mass & Lean Mass (Weight of everything except fat)
		*Available Health Status*
		- Less than Essential Fat
		- Essential Fat
		- Athletes
		- Fitness
		- Acceptable
		- Obese"
    
    """
    url = f"https://mega-fitness-calculator1.p.rapidapi.com/bfp"
    querystring = {'weight': weight, 'age': age, 'height': height, 'gender': gender, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-fitness-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_basal_metabolic_rate_bmr(weight: int, gender: str, height: int, age: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "BMR  is the number of calories required to keep your body functioning at rest."
    
    """
    url = f"https://mega-fitness-calculator1.p.rapidapi.com/bmr"
    querystring = {'weight': weight, 'gender': gender, 'height': height, 'age': age, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mega-fitness-calculator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

