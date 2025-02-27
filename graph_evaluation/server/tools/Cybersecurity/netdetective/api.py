import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query(ipaddress: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query information about an IP address."
    ipaddress: The IP address to use for the query.
Defaults to the ip address of the connecting client
        
    """
    url = f"https://netdetective.p.rapidapi.com/query"
    querystring = {}
    if ipaddress:
        querystring['ipaddress'] = ipaddress
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netdetective.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query(ipaddress: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query information about an IP address."
    ipaddress: The IP address to use for the query.
Defaults to the ip address of the connecting client
        
    """
    url = f"https://netdetective.p.rapidapi.com/query"
    querystring = {}
    if ipaddress:
        querystring['ipaddress'] = ipaddress
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netdetective.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

