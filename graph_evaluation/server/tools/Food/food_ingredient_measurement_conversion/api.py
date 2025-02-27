import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_ingredient(value: int, is_from: str, ingredient: str, to: str, numdigit: int=3, brand: str="bob's red mill", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1. Convert between Weight and Volume, 
		     e.g., 1 cup of flour is 120 gram, or  1 ounce of butter is 2 tablespoon
		
		2. Convert in the same category, e.g., 1 cup = 16 tablespoon = 48 teaspoon;  1 ounce = 28.35 gram; or 0°C = 32°F, 100°C = 212°F; or"
    numdigit: The number of digits of result. If not provided, default 2.
        brand: if not provided, default \\\\\\\\\\\\\\\"generic\\\\\\\\\\\\\\\"
        
    """
    url = f"https://food-ingredient-measurement-conversion.p.rapidapi.com/convert"
    querystring = {'value': value, 'from': is_from, 'ingredient': ingredient, 'to': to, }
    if numdigit:
        querystring['numDigit'] = numdigit
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-ingredient-measurement-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_the_unit_weight_volume_temperature(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1. Convert between Weight and Volume, 
		     e.g., 1 cup of flour is 120 gram, or  1 ounce of butter is 2 tablespoon
		
		2. Convert in the same catagory, e.g., 1 cup = 16 tablespoon = 48 teaspoon;  1 ounce = 28.35 gram; or 0°C = 32°F, 100°C = 212°F; or"
    
    """
    url = f"https://food-ingredient-measurement-conversion.p.rapidapi.com/list-units"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-ingredient-measurement-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_ingredients(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET a list of all the Ingredients available"
    
    """
    url = f"https://food-ingredient-measurement-conversion.p.rapidapi.com/list-ingredients"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-ingredient-measurement-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

