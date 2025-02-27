import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def authentication(x_rapidapi_key: str='af2f4c1c96msh1408493318963ffp195b9cjsn99554fc04ea7', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "autentication"
    
    """
    url = f"https://teste497.p.rapidapi.com/api-football-v1.p.rapidapi.com/v3/"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "teste497.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

