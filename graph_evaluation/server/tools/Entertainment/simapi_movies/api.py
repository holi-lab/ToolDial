import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def movies(t: str, r: str, type: str, y: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    t: Movie Title
        r: Response Type
        type: Get by 'Movie' or 'Person'
        y: Year of production
        
    """
    url = f"https://simapi.p.rapidapi.com/m.php?"
    querystring = {'t': t, 'r': r, 'type': type, }
    if y:
        querystring['y'] = y
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

