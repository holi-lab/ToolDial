import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def score(a: str, b: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns only the overall match score for the relationship."
    a: Object with the birthday (including time in UTC) and location of the 1st person.
The following properties are required: year, month and day.
The following properties are recommended: hour, minute, latitude and longitude.
        b: Object with the birthday (including time in UTC) and location of the 2nd person.
The following properties are required: year, month and day.
The following properties are recommended: hour, minute, latitude and longitude.
        
    """
    url = f"https://astro-matcher-api.p.rapidapi.com/score"
    querystring = {'a': a, 'b': b, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astro-matcher-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match(a: str, b: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the two horoscopes, the synastry and our rating of the relationship."
    a: Object with the birthday (including time in UTC) and location of the 1st person.
The following properties are required: year, month and day.
The following properties are recommended: hour, minute, latitude and longitude.
        b: Object with the birthday (including time in UTC) and location of the 2nd person.
The following properties are required: year, month and day.
The following properties are recommended: hour, minute, latitude and longitude.
        
    """
    url = f"https://astro-matcher-api.p.rapidapi.com/match"
    querystring = {'a': a, 'b': b, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "astro-matcher-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

