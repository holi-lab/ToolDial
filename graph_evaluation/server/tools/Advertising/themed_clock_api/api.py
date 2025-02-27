import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def theme(tz: str='US/Eastern', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API endpoint is /theme and it supports only GET method.
		Parameters
		tz : The user's preferred time zone, it should be one of the following: 'UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific', if not provided it will be set to 'UTC'"
    
    """
    url = f"https://themed-clock-api.p.rapidapi.com/theme"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "themed-clock-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

