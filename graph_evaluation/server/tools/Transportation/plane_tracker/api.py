import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def plane_info(icao: str=None, reg: str='N12345', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Info of plane like position and more."
    
    """
    url = f"https://plane-tracker.p.rapidapi.com/"
    querystring = {}
    if icao:
        querystring['icao'] = icao
    if reg:
        querystring['reg'] = reg
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plane-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

