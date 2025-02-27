import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_trivia(limit: int=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Trivia API endpoint"
    limit: How many results to return. Must be between **1** and **30**. Default is **1**.
        category: Category of trivia. Possible values are: 

**artliterature
language
sciencenature
general
fooddrink
peopleplaces
geography
historyholidays
entertainment
toysgames
music
mathematics
religionmythology
sportsleisure**
        
    """
    url = f"https://trivia-by-api-ninjas.p.rapidapi.com/v1/trivia"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trivia-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

