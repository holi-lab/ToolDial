import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(max: int=10, randomize: bool=None, category: str='all', provider: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive french news in JSON format"
    max: Limit result size. By default, the value is **30**

Values accepted : between 1 and 30
        randomize: Randomize the result (each request has different results). By default, the value is **0**

Values accepted : 0 or 1
        category: Filter news by category. By default, the value is **all**

Values accepted : 
- all
- general
- sport
- culture
- political
- miscellaneous
        provider: Filter news by provider. By default, the value is **all**

Values accepted : 
- all
- francetvinfo.fr
- europe1.fr
- lesechos.fr
- leparisien.fr
- nouvelobs.com
        
    """
    url = f"https://frenchnews.p.rapidapi.com/get"
    querystring = {}
    if max:
        querystring['max'] = max
    if randomize:
        querystring['randomize'] = randomize
    if category:
        querystring['category'] = category
    if provider:
        querystring['provider'] = provider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "frenchnews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

