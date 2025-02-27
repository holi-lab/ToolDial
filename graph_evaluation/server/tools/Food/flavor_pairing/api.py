import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ingredients(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all the available ingredients"
    
    """
    url = f"https://flavor-pairing.p.rapidapi.com/ingredients"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flavor-pairing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pairings(ingredients: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return a JSON of the pairings.
		
		**Usage**
		
		pairing?ingredients=ingredient1,ingredient2,etc.
		
		accepts only ingredients you can access from the endpoint /ingredients"
    
    """
    url = f"https://flavor-pairing.p.rapidapi.com/pairing"
    querystring = {'ingredients': ingredients, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flavor-pairing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

