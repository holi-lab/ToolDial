import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def coordinated_universal_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the coordinated universal time"
    
    """
    url = f"https://world-clock.p.rapidapi.com/json/utc/now"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-clock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_time_within_a_timezone(time_zone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Current Time for EST"
    time_zone: Timezone (see: https://www.timeanddate.com/time/zones/)
        
    """
    url = f"https://world-clock.p.rapidapi.com/json/{time_zone}/now"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-clock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jsonp(time_zone: str, callback: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current timezone (JSONP)"
    time_zone: timezone (see:https://en.wikipedia.org/wiki/List_of_time_zone_abbreviations)
        
    """
    url = f"https://world-clock.p.rapidapi.com/jsonp/{time_zone}/utc"
    querystring = {'callback': callback, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-clock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

