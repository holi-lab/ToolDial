import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def candprimeministers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information for all Canadian Prime ministers throughout Canadian history. Includes Prime minister names, political parties and years in office."
    
    """
    url = f"https://global-governmental-data.p.rapidapi.com/CANprimeministers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-governmental-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uspresidents(president_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint provides data for all US Presidents and Vice Presidents across entire U.S History"
    
    """
    url = f"https://global-governmental-data.p.rapidapi.com/USpresidents/"
    querystring = {}
    if president_name:
        querystring['president_name'] = president_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-governmental-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

