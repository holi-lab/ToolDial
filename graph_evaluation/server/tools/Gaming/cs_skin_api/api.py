import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_skin_from_weapon(weapon_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve name, price, icon, class and weapon for a random counterstrike skin from a specific weapon
		(prices for field tested condition in USD)
		
		(replace weapon_name)
		
		Example: /api/random/AWP"
    
    """
    url = f"https://cs-skin-api.p.rapidapi.com/random/{weapon_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cs-skin-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_skin_from_class(class_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve name, price, icon, class and weapon for a random counterstrike skin from a specific class 
		(prices for field tested condition in USD)
		
		(replace class_name)
		
		Example: /api/randomclass/Rifle"
    
    """
    url = f"https://cs-skin-api.p.rapidapi.com/randomclass/{class_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cs-skin-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_skin(skin_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve name, price, icon, class and weapon for a specific counterstrike skin 
		(prices for field tested condition in USD)
		
		(replace skin_name, underscores instead of spaces)
		
		Example: /api/AK-47_Redline"
    
    """
    url = f"https://cs-skin-api.p.rapidapi.com/{skin_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cs-skin-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_skin(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve name, price, icon, class and weapon for a random counterstrike skin (prices for field tested condition)"
    
    """
    url = f"https://cs-skin-api.p.rapidapi.com/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cs-skin-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

