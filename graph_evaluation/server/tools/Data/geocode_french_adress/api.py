import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocode_french_adress(adresse: str, ville: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrouver la géo position d'une adresse française.
		Retreive geo-postion of french adress"
    
    """
    url = f"https://geocode_french_adress.p.rapidapi.com/webhook/b3dd6ed0-fa96-4bfd-b6b9-7097957ddabe"
    querystring = {'adresse': adresse, 'ville': ville, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocode_french_adress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

