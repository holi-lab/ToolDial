import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def events(to: str=None, is_from: str=None, countries: str='US,GB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all economic events 
		Filter by countries symbols like: US, JP, GB and so on. You can add multiple separating by comma or remove to get all events.
		Filter by date: from & to using date in format 2023-05-09"
    
    """
    url = f"https://economic-events-calendar.p.rapidapi.com/events"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if countries:
        querystring['countries'] = countries
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "economic-events-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

