import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_recipe(ingredients: str, name: str='(Generate Name)', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Uses AI to generate a unique recipe based on a provided name and a list of ingredients"
    
    """
    url = f"https://recipe-generator2.p.rapidapi.com/Recipes/GenerateRecipe"
    querystring = {'ingredients': ingredients, }
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-generator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

