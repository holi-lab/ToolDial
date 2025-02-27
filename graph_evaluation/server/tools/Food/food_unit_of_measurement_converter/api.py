import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ingredients(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://food-unit-of-measurement-converter.p.rapidapi.com/ingredients"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-unit-of-measurement-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(unit: str, ingredient: str, value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a json containing all units of measurement"
    unit: One of `grams`, `oz`, `lbs`, `milliliters`, `cups`, `teaspoons`, `tablespoons`, `fl_oz`, `liters`, `quarts`, `pints`.
        ingredient: One of the available ingredients in /ingredients. Example: `whole_wheat_flour`
        value: Value of your unit. Example: `50`
        
    """
    url = f"https://food-unit-of-measurement-converter.p.rapidapi.com/convert"
    querystring = {'unit': unit, 'ingredient': ingredient, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-unit-of-measurement-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

