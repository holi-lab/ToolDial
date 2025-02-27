import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def female_pet_names(search: str='G', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate female pet names
		
		You can include a 'search' parameter to retrieve a randomized pet name that starts with the value of 'search'
		
		for example:
		
		/female-pet-names?search=A
		
		would retrieve a female pet name that starts with the letter A or a.
		
		Multiple letters can also be used:
		
		/female-pet-names?search=Albe
		
		would retrieve a female pet names that starts with the letters `Albe` or `albe`."
    
    """
    url = f"https://pet-name-generator.p.rapidapi.com/female-pet-names"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pet-name-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def male_pet_names(search: str='Ba', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate male pet names
		
		You can include a 'search' parameter to retrieve a randomized pet name that starts with the value of 'search'
		
		for example:
		
		/male-pet-names?search=A
		
		would retrieve a male pet name that starts with the letter A or a.
		
		Multiple letters can also be used:
		
		/male-pet-names?search=Albe
		
		would retrieve a male pet names that starts with the letters `Albe` or `albe`."
    
    """
    url = f"https://pet-name-generator.p.rapidapi.com/male-pet-names"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pet-name-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_pet_names(search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate all pet names
		
		You can include a 'search' parameter to retrieve a randomized pet name that starts with the value of 'search'
		
		for example:
		
		/all-pet-names?search=A
		
		would retrieve a name from all pet names that start with the letter A or a.
		
		Multiple letters can also be used:
		
		/all-pet-names?search=Albe
		
		would retrieve a name from all pet names that start with the letters `Albe` or `albe`."
    search: The `search` parameter is used to specifically search for names in the API.
        
    """
    url = f"https://pet-name-generator.p.rapidapi.com/all-pet-names"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pet-name-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

