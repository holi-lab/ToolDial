import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_character(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Characters: The search endpoint allows you to search for anime characters. You can specify your search query by adding a q parameter to the URL. For example, if you want to search for "Goku", the URL will look like this: /characters/search?q=goku"
    
    """
    url = f"https://animepedia.p.rapidapi.com/characters/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animepedia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

