import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def free_access(get_dash_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The api call will grant access to a unique URL to access the dashboard (demo [here](https://walnuthillai.com/wp-content/themes/blankslate-child/videos/demo_walnuttradingdash.mp4)). The FREE version grants access to half of the assets and technical indicators / strategies and any time period from 2021/06 to 2022/1."
    
    """
    url = f"https://walnuttradingdash.p.rapidapi.com/free_access"
    querystring = {'get_dash_url': get_dash_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walnuttradingdash.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def full_access(get_dash_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The api call will grant access to a unique URL to access the dashboard (demo [here](https://walnuthillai.com/wp-content/themes/blankslate-child/videos/demo_walnuttradingdash.mp4)). The PRO version grants access to 60 minutes of use per request, unlimited access to all assets and technical indicators, and any time period up to the previous date."
    
    """
    url = f"https://walnuttradingdash.p.rapidapi.com/full_access"
    querystring = {'get_dash_url': get_dash_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walnuttradingdash.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

