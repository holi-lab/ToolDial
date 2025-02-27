import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_icon(access_control_allow_origin: str='*', accept: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get icons"
    
    """
    url = f"https://xctuality-interactive.p.rapidapi.com/v3/search/icons/priority"
    querystring = {}
    if access_control_allow_origin:
        querystring['Access-Control-Allow-Origin'] = access_control_allow_origin
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xctuality-interactive.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

