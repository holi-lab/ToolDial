import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def melted_polar_ice_cap_copy(polarice: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response will be an object with no CORS resources enabled."
    
    """
    url = f"https://melted-polar-ice-cap.p.rapidapi.com/api/arctic-api"
    querystring = {}
    if polarice:
        querystring['PolarIce'] = polarice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melted-polar-ice-cap.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def melted_polar_ice_cap(polarice: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response will be an object with no CORS resources enabled."
    
    """
    url = f"https://melted-polar-ice-cap.p.rapidapi.com/api/arctic-api"
    querystring = {}
    if polarice:
        querystring['PolarIce'] = polarice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "melted-polar-ice-cap.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

