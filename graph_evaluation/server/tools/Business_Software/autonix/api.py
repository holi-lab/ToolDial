import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_locations(api_token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all locations associated with this account."
    
    """
    url = f"https://autonix.p.rapidapi.com/location"
    querystring = {'api_token': api_token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autonix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def visits_by_location(location_id: int, api_token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API gets all the visits, including visitor data, by location.  Location ID is required (available on each location here: https://autonix.io/locations"
    
    """
    url = f"https://autonix.p.rapidapi.com/visits/location/{location_id}"
    querystring = {'api_token': api_token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autonix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_usage(api_token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Monitor your usage of Autonix with this Check Usage endpoint.  Includes usage by check-in type as well as any purchased Add-ons."
    
    """
    url = f"https://autonix.p.rapidapi.com/visits/checkin-usage"
    querystring = {'api_token': api_token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autonix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

