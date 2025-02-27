import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def is_prime(number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "checks if a number is prime"
    
    """
    url = f"https://teamriverbubbles-random-utilities1.p.rapidapi.com/math/prime"
    querystring = {'number': number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teamriverbubbles-random-utilities1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kda_calculator(deaths: int, assists: int, kills: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates a kd (kill + assists death ratio)"
    
    """
    url = f"https://teamriverbubbles-random-utilities1.p.rapidapi.com/vgts/kda"
    querystring = {'deaths': deaths, 'assists': assists, 'kills': kills, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teamriverbubbles-random-utilities1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kd_calculator(kills: int, deaths: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates a kd (kill death ratio)"
    
    """
    url = f"https://teamriverbubbles-random-utilities1.p.rapidapi.com/vgts/kd"
    querystring = {'kills': kills, 'deaths': deaths, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teamriverbubbles-random-utilities1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_uuid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "creates a random uuid v4"
    
    """
    url = f"https://teamriverbubbles-random-utilities1.p.rapidapi.com/random/uuid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teamriverbubbles-random-utilities1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_skin_from_uuid(uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gets a skin from a uuid"
    
    """
    url = f"https://teamriverbubbles-random-utilities1.p.rapidapi.com/minecraft/skin/{uuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teamriverbubbles-random-utilities1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uuid_to_username(uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "turns a uuid into a username"
    
    """
    url = f"https://teamriverbubbles-random-utilities1.p.rapidapi.com/minecraft/username/{uuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teamriverbubbles-random-utilities1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def username_to_uuid(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "turns a username into a uuid"
    
    """
    url = f"https://teamriverbubbles-random-utilities1.p.rapidapi.com/minecraft/uuid/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teamriverbubbles-random-utilities1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

