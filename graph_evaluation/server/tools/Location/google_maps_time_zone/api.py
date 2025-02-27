import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def timezone(timestamp: str, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the timezone information based on a latitude-longitude location"
    timestamp: specifies the desired time as seconds since midnight, January 1, 1970 UTC. The Time Zone API uses the timestamp to determine whether or not Daylight Savings should be applied, based on the time zone of the location. Note that the API does not take historical time zones into account. That is, if you specify a past timestamp, the API does not take into account the possibility that the location was previously in a different time zone.
        location: a comma-separated lat,lng tuple (eg. location=-33.86,151.20), representing the location to look up.
        
    """
    url = f"https://google-maps-time-zone.p.rapidapi.com/timezone/json"
    querystring = {'timestamp': timestamp, 'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-maps-time-zone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

