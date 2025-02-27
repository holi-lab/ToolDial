import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def insights(m: int, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access all the processed insights in order to boost revenue on metasearch channels."
    m: Last \"m\" minutes of insights.
        token: Contact an account manager to request this for your brand.
        
    """
    url = f"https://traveldax.p.rapidapi.com/insights"
    querystring = {'m': m, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "traveldax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_search(origin: str, market: str, outbound: str, destination: str, meta: str, pax: str, currency: str, token: str, type: str, inbound: str='2022-02-12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Add a new search to the queue. It will be executed in real time."
    token: Contact an account manager to request this for your brand.
        
    """
    url = f"https://traveldax.p.rapidapi.com/new_search"
    querystring = {'origin': origin, 'market': market, 'outbound': outbound, 'destination': destination, 'meta': meta, 'pax': pax, 'currency': currency, 'token': token, 'type': type, }
    if inbound:
        querystring['inbound'] = inbound
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "traveldax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

