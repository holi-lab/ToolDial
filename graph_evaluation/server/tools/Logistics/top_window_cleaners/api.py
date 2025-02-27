import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_window_cleaners(windowcleanerslondon: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Professional window cleaning in London"
    
    """
    url = f"https://top-window-cleaners.p.rapidapi.com/"
    querystring = {}
    if windowcleanerslondon:
        querystring['windowcleanerslondon'] = windowcleanerslondon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "top-window-cleaners.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

