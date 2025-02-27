import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def makes_lookup(format: str='json', key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the makes endpoint to request a list of makes for a given year."
    
    """
    url = f"https://windshieldrepairtech-windshield-repair-tech-auto-glass-v1.p.rapidapi.com/vehicle/v1/makes/"
    querystring = {}
    if format:
        querystring['format'] = format
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "windshieldrepairtech-windshield-repair-tech-auto-glass-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_lookup(format: str='json', key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the years endpoint to request a list of years that vehicles were produced in."
    
    """
    url = f"https://windshieldrepairtech-windshield-repair-tech-auto-glass-v1.p.rapidapi.com/vehicle/v1/years/"
    querystring = {}
    if format:
        querystring['format'] = format
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "windshieldrepairtech-windshield-repair-tech-auto-glass-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

