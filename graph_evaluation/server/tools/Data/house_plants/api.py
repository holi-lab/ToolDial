import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbysearch(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches through all plants and returns the closest results matching "Latin name", "Family", "Other names", "Common name", "Common name (fr.)", "Description", "Categories", "Origin" in order by highest most likely. This route is case insensitive and uses fuzzy search"
    
    """
    url = f"https://house-plants2.p.rapidapi.com/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "house-plants2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getall(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all plant data within an array. Information for each plant returned contains (but not limited to) the list below
		
		- Latin name 
		- Img 
		- Id
		- Family 
		- Other names 
		- Common name 
		- Common name (fr.) 
		- Description 
		- Categories 
		- Origin 
		- Climat 
		- Temperature max
		- Temperature min
		- Zone 
		- Growth 
		- Light ideal 
		- Light tolered 
		- Watering 
		- Insects 
		- Disease 
		- Appeal 
		- Color of leaf 
		- Color of blooms 
		- Blooming season 
		- Perfume 
		- Avaibility 
		- Pot diameter
		- Height at purchase
		- Width at purchase
		- Height potential
		- Width potential
		- Available sizes
		- Bearing 
		- Clump
		- Pruning 
		- Style 
		- Use"
    
    """
    url = f"https://house-plants2.p.rapidapi.com/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "house-plants2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getalllite(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all items but only identifying data. All additional data can be grabbed from the GetById route or the GetAll route which return all item with all its keys, no restriction"
    
    """
    url = f"https://house-plants2.p.rapidapi.com/all-lite"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "house-plants2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallcategories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available categories for all plants"
    
    """
    url = f"https://house-plants2.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "house-plants2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbyid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single plants data within an object. Information for each plant returned contains (but not limited to) the list below
		
		- Latin name 
		- Img 
		- Id
		- Family 
		- Other names 
		- Common name 
		- Common name (fr.) 
		- Description 
		- Categories 
		- Origin 
		- Climat 
		- Temperature max
		- Temperature min
		- Zone 
		- Growth 
		- Light ideal 
		- Light tolered 
		- Watering 
		- Insects 
		- Disease 
		- Appeal 
		- Color of leaf 
		- Color of blooms 
		- Blooming season 
		- Perfume 
		- Avaibility 
		- Pot diameter
		- Height at purchase
		- Width at purchase
		- Height potential
		- Width potential
		- Available sizes
		- Bearing 
		- Clump
		- Pruning 
		- Style 
		- Use"
    
    """
    url = f"https://house-plants2.p.rapidapi.com/id/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "house-plants2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbycategory(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all item based on a category"
    
    """
    url = f"https://house-plants2.p.rapidapi.com/category/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "house-plants2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

