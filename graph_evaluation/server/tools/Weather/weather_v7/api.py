import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def los_angeles_city_copy(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical weather for any location
		Our new technology, Time Machine, has allowed us to enhance the data in the Historical Weather Collection.
		
		Historical weather data available for ANY coordinate
		The depth of historical data have been extended to 40 YEARS
		You can download data from Personal account or contact us to order it."
    
    """
    url = f"https://weather1493.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather1493.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def los_angeles_city(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical weather for any location
		Our new technology, Time Machine, has allowed us to enhance the data in the Historical Weather Collection.
		
		Historical weather data available for ANY coordinate
		The depth of historical data have been extended to 40 YEARS
		You can download data from Personal account or contact us to order it."
    
    """
    url = f"https://weather1493.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather1493.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

