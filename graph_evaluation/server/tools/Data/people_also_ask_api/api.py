import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def people_also_ask(query: str, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract "People also ask" and "Related searches""
    country: ISO 3166-1 alpha-2 code (us, gb, de, ....)
        
    """
    url = f"https://people-also-ask-api.p.rapidapi.com/people_also_ask"
    querystring = {'query': query, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "people-also-ask-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

