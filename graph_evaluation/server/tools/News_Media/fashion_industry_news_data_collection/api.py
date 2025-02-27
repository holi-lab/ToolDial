import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_term(q: str, tsi: int=1677067077000, ts: int=1675159335000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search term"
    tsi: This is the final time delimiter. Unix Time format in milliseconds.

Now default.
        ts: Initial date-time limit reference in Unix time (miliseconds)

1 month ago by default
        
    """
    url = f"https://fashion-industry-news-data-collection.p.rapidapi.com/"
    querystring = {'q': q, }
    if tsi:
        querystring['tsi'] = tsi
    if ts:
        querystring['ts'] = ts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fashion-industry-news-data-collection.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

