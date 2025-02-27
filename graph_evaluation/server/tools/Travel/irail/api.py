import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def departures(stationname: str, year: str, month: str, day: str, hour: str, minutes: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Departures of trains in Belgium"
    stationname: You can find the appropriate names in the Stations list
        year: The year you want to query
        month: The number of the month
        day: The number of the day
        hour: The number of hours in 24h style
        minutes: The minutes you want to retrieve responses from
        
    """
    url = f"https://irail.p.rapidapi.com/NMBS/Departures/{stationname}/{year}/{month}/{day}/{hour}/{minutes}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://irail.p.rapidapi.com/NMBS/Stations.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

