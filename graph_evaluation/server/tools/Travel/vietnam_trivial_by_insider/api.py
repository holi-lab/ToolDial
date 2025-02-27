import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vietnam_statistics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full statistics about the country Vietnam"
    
    """
    url = f"https://vietnam-trivial-by-insider.p.rapidapi.com/vietnam_statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vietnam-trivial-by-insider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vietnam_fun_facts_for_travellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API contains facts and trivial from a native born Vietnamese who often surprised at funny mistakes travelers made while visiting Vietnam.  Enjoy!"
    
    """
    url = f"https://vietnam-trivial-by-insider.p.rapidapi.com/vietnam_trivials"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vietnam-trivial-by-insider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

