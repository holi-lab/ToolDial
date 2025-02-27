import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_suburbs_and_postcodes_in_a_radius(lat: str, radius: int, lng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes your epic centre latitude, longitude along with radius in KM and returns all postcodes and suburbs within it."
    
    """
    url = f"https://australian-postcode-to-suburb.p.rapidapi.com/radius/{lat}/{lng}/{radius}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-postcode-to-suburb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_states(postcode: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return 3 letter states for the passed postcode."
    postcode: Use a valid Australian postcode.
        
    """
    url = f"https://australian-postcode-to-suburb.p.rapidapi.com/states/{postcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-postcode-to-suburb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_suburbs(postcode: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all matching suburbs for the passed postcode. The response also includes the state for each suburb and, if available, latitude and longitude for the suburb."
    
    """
    url = f"https://australian-postcode-to-suburb.p.rapidapi.com/suburbs/{postcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-postcode-to-suburb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

