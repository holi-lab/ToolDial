import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_recipe(ingredient: str='chicken', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate your recipe for your use case or application!"
    
    """
    url = f"https://generate-a-recipe-based-on-an-ingredient.p.rapidapi.com/generate_recipe"
    querystring = {}
    if ingredient:
        querystring['ingredient'] = ingredient
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "generate-a-recipe-based-on-an-ingredient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

