import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nz_plate_decode(platenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "NZ Plate Decode"
    
    """
    url = f"https://plate-number-decoder.p.rapidapi.com/b2b/plate-precheck/NZ"
    querystring = {'plateNumber': platenumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plate-number-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def za_plate_decode(platenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ZA Plate Decode"
    
    """
    url = f"https://plate-number-decoder.p.rapidapi.com/b2b/plate-precheck/ZA"
    querystring = {'plateNumber': platenumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plate-number-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def au_plate_decode(platenumber: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AU Plate Decode"
    
    """
    url = f"https://plate-number-decoder.p.rapidapi.com/b2b/plate-precheck/AU"
    querystring = {'plateNumber': platenumber, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plate-number-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

