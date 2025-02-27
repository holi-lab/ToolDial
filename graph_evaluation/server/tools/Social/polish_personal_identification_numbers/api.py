import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def passport_check(number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checking and returning informations about given passport number."
    number: Passport document number. * REQUIRED *
        
    """
    url = f"https://verifeo.p.rapidapi.com/passport/check"
    querystring = {'number': number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "verifeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pesel_multicheck(numbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checking and returning informations about each of given PESEL number."
    numbers: Coma separated, PESEL numbers. * REQUIRED *
        
    """
    url = f"https://verifeo.p.rapidapi.com/pesel/multicheck"
    querystring = {'numbers': numbers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "verifeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def passport_multicheck(numbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checking and returning informations about each of given passport number."
    numbers: Coma separated, passport documents numbers. * REQUIRED *
        
    """
    url = f"https://verifeo.p.rapidapi.com/passport/multicheck"
    querystring = {'numbers': numbers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "verifeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

