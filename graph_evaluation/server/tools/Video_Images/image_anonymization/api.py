import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_available_modes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Service provides server modes that may be used to choose which kind of objects to hide. This endpoint returns list of modes that may be used as query parameters for requests to the `results` endpoint.
		
		The following modes are supported:
		
		* `hide-clp` – to hide car license plates
		* `hide-face` – to hide faces"
    
    """
    url = f"https://image-anonymization.p.rapidapi.com/v1/modes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-anonymization.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an actual version of the service in format `vX.Y.Z` where X is the version of API."
    
    """
    url = f"https://image-anonymization.p.rapidapi.com/v1/version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-anonymization.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

