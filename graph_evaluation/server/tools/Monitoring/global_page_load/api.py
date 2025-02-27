import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def global_page_load(origin: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides performance measurement for a Web page or API call from a specified geographic location"
    origin: Geographic Location
        url: Url of page or asset to be measured
        
    """
    url = f"https://global-page-load.p.rapidapi.com/globalpageload"
    querystring = {'origin': origin, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-page-load.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

