import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_followme_uploadfileanonymously(https_rapidapi_com: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The aim of the FollowMe Tracking Servie API is to enable developers on brining our tracking services to more convenient and accessible platforms for users"
    
    """
    url = f"https://fm_api_release-v0-1.p.rapidapi.com{https_rapidapi_com}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fm_api_release-v0-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

