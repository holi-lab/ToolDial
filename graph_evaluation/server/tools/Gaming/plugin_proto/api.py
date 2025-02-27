import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_points(iniyethawe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For unlimited lucky"
    iniyethawe: -1
        iniyethawe: -1
        
    """
    url = f"https://plugin-proto.p.rapidapi.com/proto/api/token"
    querystring = {'iniyethawe': iniyethawe, }
    if iniyethawe:
        querystring['iniyethawe'] = iniyethawe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plugin-proto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

