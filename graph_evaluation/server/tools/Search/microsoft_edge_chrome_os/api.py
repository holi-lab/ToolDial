import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def micro_edge_copy(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gybghbjnj"
    
    """
    url = f"https://microsoft-edge-chrome-os.p.rapidapi.com/fa8d5da0-595a-11ec-a160-7f7a48c20b2d"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "microsoft-edge-chrome-os.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def micro_edge(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gybghbjnj"
    
    """
    url = f"https://microsoft-edge-chrome-os.p.rapidapi.com/fa8d5da0-595a-11ec-a160-7f7a48c20b2d"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "microsoft-edge-chrome-os.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

