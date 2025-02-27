import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def metric_kilograms(weight: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will calculate the BMI of an individual using the imperial measurement system."
    weight: The weight of someone in kilograms (kgs)
        height: The height of someone in meters (m)
        
    """
    url = f"https://body-mass-index-bmi-calculator.p.rapidapi.com/metric"
    querystring = {'weight': weight, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "body-mass-index-bmi-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def imperial_pounds(weight: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will calculate the BMI of an individual using the imperial measurement system."
    weight: The weight of someone in pounds (lbs) 
        height: The height of someone in inches (in)
        
    """
    url = f"https://body-mass-index-bmi-calculator.p.rapidapi.com/imperial"
    querystring = {'weight': weight, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "body-mass-index-bmi-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weight_category(bmi: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the weight category of an individual based on their BMI"
    
    """
    url = f"https://body-mass-index-bmi-calculator.p.rapidapi.com/weight-category"
    querystring = {'bmi': bmi, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "body-mass-index-bmi-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

