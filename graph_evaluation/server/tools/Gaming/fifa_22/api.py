import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def players(limit: int, offset: int, filter: str='Ronaldo', sort: str='ranking:ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Players Stats"
    
    """
    url = f"https://fifa-222.p.rapidapi.com/v2/entities/fifa-22-ratings"
    querystring = {'limit': limit, 'offset': offset, }
    if filter:
        querystring['filter'] = filter
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-222.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

