import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def street(street: str, topics: str='history:1,tourism:1', temperature: str='0', style: str='audioguide', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "StreetNarrator API Get a unique history/touristic AI text for any street around the world"
    
    """
    url = f"https://streetnarrator.p.rapidapi.com/prod/street"
    querystring = {'street': street, }
    if topics:
        querystring['topics'] = topics
    if temperature:
        querystring['temperature'] = temperature
    if style:
        querystring['style'] = style
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streetnarrator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

