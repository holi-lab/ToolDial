import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detect_nudity_in_web_hosted_image(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect Nudity in Web-Hosted Image"
    
    """
    url = f"https://netspark-nude-detect-v1.p.rapidapi.com/url/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netspark-nude-detect-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect_illegal_web_hosted_image(uri: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect image category from:
		very likely nude, likely nude, very likely minimal dress, likely minimal dress, very likely partial dress, full dress, likely partial dress, men and objects, possible pedophilic, likely pedophilic, very likely pedophilic"
    
    """
    url = f"https://netspark-nude-detect-v1.p.rapidapi.com/uri/"
    querystring = {'uri': uri, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netspark-nude-detect-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

