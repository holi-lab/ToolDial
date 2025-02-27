import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def powerful_female_marvel_superheroes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most Powerful Female Marvel Superheroes"
    
    """
    url = f"https://powerful-female-marvel-superheroes.p.rapidapi.com/vyvPl8/powerfulfemalemarvelsuperheroes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "powerful-female-marvel-superheroes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

