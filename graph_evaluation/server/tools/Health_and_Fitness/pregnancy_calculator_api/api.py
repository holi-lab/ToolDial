import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fertility_window_get(menstrual_date: str, cycle_length: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint accepts a GET request to calculate the fertility window."
    menstrual_date: The date of the first day of the last menstrual period in the format '**YYYY-MM-DD**'.
        cycle_length: The length of the menstrual cycle in days.
        
    """
    url = f"https://pregnancy-calculator-api.p.rapidapi.com/fw"
    querystring = {'menstrual_date': menstrual_date, 'cycle_length': cycle_length, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pregnancy-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pregnancy_weight_recommendation_get(pre_pregnancy_weight: int, gestational_age: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint that accepts a GET request to calculate the recommended weight during pregnancy based on the user's pre-pregnancy weight, height, and current gestational age."
    pre_pregnancy_weight: The user's pre-pregnancy weight in kilograms (**kg**).
        gestational_age: The current gestational age in **weeks**.

        height: The user's height in meters (**m**).
        
    """
    url = f"https://pregnancy-calculator-api.p.rapidapi.com/pwr"
    querystring = {'pre_pregnancy_weight': pre_pregnancy_weight, 'gestational_age': gestational_age, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pregnancy-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conception_date(conception_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the current week of pregnancy based on the conception date."
    conception_date: The date of conception in the format '**YYYY-MM-DD**'
        
    """
    url = f"https://pregnancy-calculator-api.p.rapidapi.com/pw/conception"
    querystring = {'conception_date': conception_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pregnancy-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_menstrual_period_lmp(cycle_length: int, last_period_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the current week of pregnancy based on the Last Menstrual Period (LMP)."
    cycle_length: The average length of the user's menstrual cycle in days.
        last_period_date: The date of the user's last menstrual period in the format '**YYYY-MM-DD**'.
        
    """
    url = f"https://pregnancy-calculator-api.p.rapidapi.com/pw/lmp"
    querystring = {'cycle_length': cycle_length, 'last_period_date': last_period_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pregnancy-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conception_date(conception_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the estimated due date based on the user's conception date."
    conception_date: The date of conception in the format '**YYYY-MM-DD**'
        
    """
    url = f"https://pregnancy-calculator-api.p.rapidapi.com/dd/conception"
    querystring = {'conception_date': conception_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pregnancy-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_menstrual_period_lmp(cycle_length: str, last_period_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the estimated due date based on the user's last menstrual period (LMP)"
    cycle_length: The average length of the user's menstrual cycle in days.
        last_period_date: The date of the user's last menstrual period in the format '**YYYY-MM-DD**'.
        
    """
    url = f"https://pregnancy-calculator-api.p.rapidapi.com/dd/lmp"
    querystring = {'cycle_length': cycle_length, 'last_period_date': last_period_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pregnancy-calculator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

