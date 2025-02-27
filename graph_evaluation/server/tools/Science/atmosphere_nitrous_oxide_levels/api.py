import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nitrous_endpoint(nitrous: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response will be an object with no CORS resources enabled."
    
    """
    url = f"https://atmosphere-nitrous-oxide-levels.p.rapidapi.com/api/nitrous-oxide-api"
    querystring = {}
    if nitrous:
        querystring['nitrous'] = nitrous
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "atmosphere-nitrous-oxide-levels.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

