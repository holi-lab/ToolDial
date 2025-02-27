import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def real_estate_api(house1: str='1', house2: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Real Estate"
    
    """
    url = f"https://real_estate_heroku.p.rapidapi.com/House1"
    querystring = {}
    if house1:
        querystring['House1'] = house1
    if house2:
        querystring['House2'] = house2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real_estate_heroku.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

