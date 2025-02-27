import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tor_exit_nodes_geolocated(order: str=None, localitylanguage: str='en', offset: int=0, sort: str=None, batchsize: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the list of active Tor exit nodes geolocated to country-level along with active carrier information."
    
    """
    url = f"https://tor-exit-nodes-geolocated.p.rapidapi.com/data/tor-exit-nodes-list"
    querystring = {}
    if order:
        querystring['order'] = order
    if localitylanguage:
        querystring['localityLanguage'] = localitylanguage
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if batchsize:
        querystring['batchSize'] = batchsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tor-exit-nodes-geolocated.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

