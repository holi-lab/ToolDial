import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_quotes(name: str='alex', search: str='exercise', tag: str='crossfit', limit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves quotes according to the selected parameters."
    
    """
    url = f"https://bulk-hackers-quotes.p.rapidapi.com/wp-json/bulk-hackers/get-quotes"
    querystring = {}
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if tag:
        querystring['tag'] = tag
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bulk-hackers-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

