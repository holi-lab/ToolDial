import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ultimate(ultimate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query information about fifth-generation fighter-jets."
    
    """
    url = f"https://jetionay-pro.p.rapidapi.com/ultimate"
    querystring = {}
    if ultimate:
        querystring['ultimate'] = ultimate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jetionay-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def elite(elite: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query information about fourth-generation fighter-jets."
    
    """
    url = f"https://jetionay-pro.p.rapidapi.com/elite"
    querystring = {}
    if elite:
        querystring['elite'] = elite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jetionay-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced(advanced: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query information about third-generation fighter-jets."
    
    """
    url = f"https://jetionay-pro.p.rapidapi.com/advanced"
    querystring = {}
    if advanced:
        querystring['advanced'] = advanced
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jetionay-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def intermediate(intermediate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query information about second-generation fighter-jets."
    
    """
    url = f"https://jetionay-pro.p.rapidapi.com/intermediate"
    querystring = {}
    if intermediate:
        querystring['intermediate'] = intermediate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jetionay-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fundamentals(fundamentals: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query information about first-generation fighter-jets."
    
    """
    url = f"https://jetionay-pro.p.rapidapi.com/fundamentals"
    querystring = {}
    if fundamentals:
        querystring['fundamentals'] = fundamentals
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jetionay-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

