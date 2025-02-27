import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_query(endtime: int, device: str, starttime: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the endpoint used to query your data points back from the API. It uses query string parameters to specify the conditions for the results to return."
    endtime: Epoch time representation of the end time of the query window
        device: The ID of the device you're querying data points for. This is a combination of 8 alphanumeric characters. Reference the /device endpoint for more information on your current devices or to add new ones.
        starttime: Epoch time representation of the start of the query window
        
    """
    url = f"https://arktis-gps-data-point-history.p.rapidapi.com/query"
    querystring = {'endTime': endtime, 'device': device, 'startTime': starttime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arktis-gps-data-point-history.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_device(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all available devices under the current user."
    
    """
    url = f"https://arktis-gps-data-point-history.p.rapidapi.com/device"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arktis-gps-data-point-history.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

