import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def summery_copy(date: str, is_id: int, value: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dahsboard summery"
    date: date description
        id: id descrition
        value: value description
        
    """
    url = f"https://summery.p.rapidapi.com/reveneus"
    querystring = {'date': date, 'id': is_id, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summery.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summery(proxysecret: int, xrapidapiproxysecret: str, xrapidapikey: str, is_id: int, date: str, value: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dahsboard summery"
    proxysecret: key description
        xrapidapiproxysecret: key description
        xrapidapikey: key description
        id: id descrition
        date: date description
        value: value description
        
    """
    url = f"https://summery.p.rapidapi.com/reveneus"
    querystring = {'ProxySecret': proxysecret, 'XRapidAPIProxySecret': xrapidapiproxysecret, 'xrapidapikey': xrapidapikey, 'id': is_id, 'date': date, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summery.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

