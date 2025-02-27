import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ppsr_lookup_by_vin(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup a PPSR by a vehicle's VIN and retrieve the search results and a PDF URL."
    
    """
    url = f"https://ppsr-search-certificate-motor-vehicle1.p.rapidapi.com/rapid_ppsrLookupVIN"
    querystring = {'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppsr-search-certificate-motor-vehicle1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_valid_vin(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Confirm that the VIN is a currently registered vehicle.  Returns the vehicle's details."
    
    """
    url = f"https://ppsr-search-certificate-motor-vehicle1.p.rapidapi.com/rapid_checkReg"
    querystring = {'VIN': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppsr-search-certificate-motor-vehicle1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ppsr_lookup_by_registration(state: str, reg: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup a PPSR by a vehicle's registration and retrieve the search results and a PDF URL.  Simple send the vehicle's number plate and state abbreviation."
    
    """
    url = f"https://ppsr-search-certificate-motor-vehicle1.p.rapidapi.com/rapid_ppsrLookup"
    querystring = {'state': state, 'reg': reg, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppsr-search-certificate-motor-vehicle1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_valid_registration(reg: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Confirm that the registration and state is a currently registered vehicle.  Returns the vehicle's details."
    
    """
    url = f"https://ppsr-search-certificate-motor-vehicle1.p.rapidapi.com/rapid_checkReg"
    querystring = {'reg': reg, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppsr-search-certificate-motor-vehicle1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

