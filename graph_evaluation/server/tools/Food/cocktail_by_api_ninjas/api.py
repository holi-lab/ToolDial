import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_cocktail(name: str='bloody mary', ingredients: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Cocktail API endpoint. Either **name** or **ingredients** parameter must be set."
    name: name of cocktail. This parameter supports partial matches (e.g. bloody will match bloody mary and bloody margarita)
        ingredients: comma-separated string of ingredients to search. Only cocktails containing all listed ingredients will be returned. For example, to search cocktails containing Vodka and lemon juice, use vodka,lemon juice.
        
    """
    url = f"https://cocktail-by-api-ninjas.p.rapidapi.com/v1/cocktail"
    querystring = {}
    if name:
        querystring['name'] = name
    if ingredients:
        querystring['ingredients'] = ingredients
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cocktail-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

