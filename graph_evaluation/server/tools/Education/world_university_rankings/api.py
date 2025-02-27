import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def university_rankings(filterbycountry: str='USA', size: int=1, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns rankings of world universities"
    
    """
    url = f"https://world-university-rankings.p.rapidapi.com/university/rankings"
    querystring = {}
    if filterbycountry:
        querystring['filterByCountry'] = filterbycountry
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-university-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

