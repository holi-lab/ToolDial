import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def disclaimer(https_mrauto_parts: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "we do not guarantee  product safety or prices"
    
    """
    url = f"https://mrautoparts.p.rapidapi.comhttp://mrauto.parts"
    querystring = {}
    if https_mrauto_parts:
        querystring['https://mrauto.parts'] = https_mrauto_parts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mrautoparts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscribe(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "add email for additional savings"
    
    """
    url = f"https://mrautoparts.p.rapidapi.comGET"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mrautoparts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def http_mrauto_parts(mrauto_parts: int=None, get: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "DOWNLOADS APP"
    get: ADD MR AUTO PARTS
        
    """
    url = f"https://mrautoparts.p.rapidapi.comhttps://mrauto.parts"
    querystring = {}
    if mrauto_parts:
        querystring['MRAUTO.PARTS'] = mrauto_parts
    if get:
        querystring['GET'] = get
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mrautoparts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

