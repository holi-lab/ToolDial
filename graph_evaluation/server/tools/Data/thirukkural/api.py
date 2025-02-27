import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def athigaram(chapternumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Thirukkural (Couplet) from an Athigaram (Chapter) by providing a chapter number"
    
    """
    url = f"https://thirukkural2.p.rapidapi.com/api/athigaram/{chapternumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thirukkural2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def thirukkural(coupletnumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Thirukkural (Couplet) by providing the couplet number"
    coupletnumber: The number should be between 1 to 1330
        
    """
    url = f"https://thirukkural2.p.rapidapi.com/api/thirukkural/{coupletnumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thirukkural2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

