import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def usa_borders_waiting_times(country: str=None, portname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all usa ports with description and wait times."
    
    """
    url = f"https://borders.p.rapidapi.com/"
    querystring = {}
    if country:
        querystring['country'] = country
    if portname:
        querystring['portName'] = portname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "borders.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

