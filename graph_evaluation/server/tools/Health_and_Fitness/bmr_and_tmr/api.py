import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bmr_index(inimperial: str, sex: str, age: int, weight: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint allows you to calculate the Basal Metabolic Rate"
    sex: Enter the gender of the person you are calculating the index for. The currently supported genders are female and male.
        age: Enter the age of the person you are calculating the indicator for.
        weight: Enter the weight of the person you are calculating the index for. The weight should be expressed in kilograms.
        height: Enter the height of the person you are calculating the index for. Height should be expressed in centimeters.
        
    """
    url = f"https://bmr-and-tmr.p.rapidapi.com/calculate-bmr"
    querystring = {'inImperial': inimperial, 'sex': sex, 'age': age, 'weight': weight, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmr-and-tmr.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tmr_index(inimperial: str, weight: int, height: int, sex: str, age: int, activity: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint allows you to calculate the Total Metabolic Rate"
    weight: Enter the weight of the person you are calculating the index for. The weight should be expressed in kilograms.
        height: Enter the height of the person you are calculating the index for. The weight should be expressed in kilograms.
        sex: Enter the gender of the person you are calculating the index for. The currently supported genders are female and male.
        age: Enter the age of the person you are calculating the index for.
        activity: Enter the physical activity of the person for whom you are calculating the index.
        
    """
    url = f"https://bmr-and-tmr.p.rapidapi.com/calculate-tmr"
    querystring = {'inImperial': inimperial, 'weight': weight, 'height': height, 'sex': sex, 'age': age, 'activity': activity, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmr-and-tmr.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def activity_values(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of possible activity values that are needed to compute TMR"
    
    """
    url = f"https://bmr-and-tmr.p.rapidapi.com/activity-values"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmr-and-tmr.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sexes_values(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of possible sexes values that are needed to compute BMR and TMR"
    
    """
    url = f"https://bmr-and-tmr.p.rapidapi.com/sex-values"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmr-and-tmr.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

