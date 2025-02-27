import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vin_detection(vin_identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is capable of resolving a Tesla VIN identifier to an actual VIN. This conversion has a success rate of approximately 98%.
		The following information is encoded in a Tesla VIN:
		Manufacturer ID
		Model Type
		Platform Type
		Seat Belt Type
		Drive System 
		Motor Type
		Year of Manufacturing
		Manufacturing Location (Fremont, Shanghai, Berlin, Austin)
		Special Series (like R for research)
		VIN Serial"
    
    """
    url = f"https://tesla-vin-identifier.p.rapidapi.com/vin/{vin_identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tesla-vin-identifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manufacturing_location_detection(vin_identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint resolves a VIN identifier to manufacturing location.
		A few examples:
		LRW-C (Shanghai, China)
		XP7-B (Berlin, Germany)"
    
    """
    url = f"https://tesla-vin-identifier.p.rapidapi.com/location/{vin_identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tesla-vin-identifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manufacturing_year_detection(vin_identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detects the manufacturing year of the respective vehicle."
    
    """
    url = f"https://tesla-vin-identifier.p.rapidapi.com/year/{vin_identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tesla-vin-identifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def model_detection(vin_identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint resolves a VIN identifier to a model type"
    
    """
    url = f"https://tesla-vin-identifier.p.rapidapi.com/s3xy/{vin_identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tesla-vin-identifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

