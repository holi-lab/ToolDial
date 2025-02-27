import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_celebrity(name: str='Michael Jordan', min_height: str=None, min_net_worth: int=None, nationality: str=None, max_net_worth: int=None, max_height: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Celebrity API endpoint."
    name: Name of the celebrity you wish to search. This field is case-insensitive.
        min_height: Minimum height of celebrities in meters (e.g. **1.65**).
        min_net_worth: Minimum net worth of celebrities.
        nationality: Nationality of celebrities. Must be an ISO 3166 Alpha-2 country code (e.g. **US**).
        max_net_worth: Maximum net worth of celebrities.
        max_height: Maximum height of celebrities in meters (e.g. **1.80**).
        
    """
    url = f"https://celebrity-by-api-ninjas.p.rapidapi.com/v1/celebrity"
    querystring = {}
    if name:
        querystring['name'] = name
    if min_height:
        querystring['min_height'] = min_height
    if min_net_worth:
        querystring['min_net_worth'] = min_net_worth
    if nationality:
        querystring['nationality'] = nationality
    if max_net_worth:
        querystring['max_net_worth'] = max_net_worth
    if max_height:
        querystring['max_height'] = max_height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "celebrity-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

