import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cocktails(ingredients: str='["rum", "whiskey", "orange juice"]', name: str='mojito', rating: int=8, iba: bool=None, description: str='made with rum', random: int=3, categories: str='["classic", "old_school"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch well known cocktails & drinks from all over the world"
    
    """
    url = f"https://cocktails.p.rapidapi.com/api/v1/cocktails"
    querystring = {}
    if ingredients:
        querystring['ingredients'] = ingredients
    if name:
        querystring['name'] = name
    if rating:
        querystring['rating'] = rating
    if iba:
        querystring['iba'] = iba
    if description:
        querystring['description'] = description
    if random:
        querystring['random'] = random
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cocktails.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

