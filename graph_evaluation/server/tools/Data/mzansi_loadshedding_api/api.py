import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def suburb(suburb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Get suburb Schedule
		Returns the schedule for the specified suburb as well as extra details about loadshedding the specified palce. Such as upcoming stages and breakdowns of power plants"
    
    """
    url = f"https://mzansi-loadshedding-api.p.rapidapi.com/schedule/{suburb}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mzansi-loadshedding-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def town(town: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## town
		Returns the town information such as province and block"
    
    """
    url = f"https://mzansi-loadshedding-api.p.rapidapi.com/{town}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mzansi-loadshedding-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

