import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, safesearch: str='0', categories: str='general', time_range: str='None', language: str='de-DE', queryid: str='0x02cAce04c8469580A2ADc20F57E143a3693c22bF', url: str='https://engine.corrently.cloud/webhook-test/44a17c5c-5087-483f-8398-dedbc8d713d4', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search endpoint - Parameters compatible to SEARXNG  (see linked external documentation). 
		In addition, a parameter `url` might be given to specify a webhook URL (POST request will be sent)."
    
    """
    url = f"https://searchhook.p.rapidapi.com/search"
    querystring = {'q': q, }
    if safesearch:
        querystring['safesearch'] = safesearch
    if categories:
        querystring['categories'] = categories
    if time_range:
        querystring['time_range'] = time_range
    if language:
        querystring['language'] = language
    if queryid:
        querystring['queryId'] = queryid
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "searchhook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webhooks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of saved queries with associated  webhooks."
    
    """
    url = f"https://searchhook.p.rapidapi.com/webhooks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "searchhook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extend(queryid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extend expiry of query identified with parameter `queryId` ."
    
    """
    url = f"https://searchhook.p.rapidapi.com/extend"
    querystring = {'queryId': queryid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "searchhook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

