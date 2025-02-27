import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def copy_of_endpoint_soccer_sports_open_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Sports Open Data"
    
    """
    url = f"https://sport1.p.rapidapi.comhttps://docs.rapidapi.com/docs/keys?a236f20115msh22fb5936ab91947p1f8113jsn65f273848924"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_sports_open_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Sports Open Data"
    
    """
    url = f"https://sport1.p.rapidapi.comhttps://docs.rapidapi.com/docs/keys?a236f20115msh22fb5936ab91947p1f8113jsn65f273848924"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

