import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gives_the_weight_catergory_based_on_the_bmi(bmienter: int, asian: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API catergorizes the person based on the BMI. The catergories would be Underweight, Normal weight, Overweight, Obesity Class I, Obesity Class II and Obesity Class III. These definitions differ in Asians hence you may be able to enter the ethnicity of the person (Asian or not asian)"
    
    """
    url = f"https://bmi10.p.rapidapi.com/bmiint"
    querystring = {'bmienter': bmienter, }
    if asian:
        querystring['asian'] = asian
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmi10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gives_the_bmi_when_you_input_height_in_feet_and_inches_input_weight_in_kilograms(weightinkg: int, heightfeet: int, heightinches: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API generates the BMI in kg/m2. You need to enter the weight in kilograms and height in feet and inches"
    
    """
    url = f"https://bmi10.p.rapidapi.com/bmicommon"
    querystring = {'weightinkg': weightinkg, 'heightfeet': heightfeet, 'heightinches': heightinches, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmi10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gives_the_bmi_when_you_input_values_in_imperial_units_feet_inches_pounds(weightinpounds: int, heightinches: int, heightfeet: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API generates the BMI in kg/m2 when the height is entered in feet and inches and weight is entered in pounds."
    
    """
    url = f"https://bmi10.p.rapidapi.com/bmi-imperial"
    querystring = {'weightinpounds': weightinpounds, 'heightinches': heightinches, 'heightfeet': heightfeet, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmi10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gives_the_bmi_when_you_input_values_in_metric_units(heightincm: int, weightinkg: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives the BMI in kg/m2 when we input height in centimeters and weight in kilograms"
    
    """
    url = f"https://bmi10.p.rapidapi.com/bmimet"
    querystring = {'heightincm': heightincm, 'weightinkg': weightinkg, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bmi10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

