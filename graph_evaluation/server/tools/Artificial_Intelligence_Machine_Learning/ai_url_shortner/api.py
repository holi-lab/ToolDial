import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shorten(shorten: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will post your shorten URL as a short Link."
    shorten: This endpoint will return the shorten URL as a short Link
        
    """
    url = f"https://ai-url-shortner.p.rapidapi.com/shorten"
    querystring = {}
    if shorten:
        querystring['shorten'] = shorten
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-url-shortner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_url(send: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to send the long URL."
    send: Use this endpoint to send the long URL
        
    """
    url = f"https://ai-url-shortner.p.rapidapi.com/send"
    querystring = {}
    if send:
        querystring['send'] = send
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-url-shortner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def welcome(welcome: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint just welcomes everyone with Hello World."
    welcome: This endpoint is just a simple hello world welcome.
        
    """
    url = f"https://ai-url-shortner.p.rapidapi.com/welcome"
    querystring = {}
    if welcome:
        querystring['welcome'] = welcome
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-url-shortner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

