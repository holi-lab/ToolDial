import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def preview_getdownloadpath(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate Limits: 100 requests per minute
		            "
    id: The unique id of the preview.
        
    """
    url = f"https://landr-mastering-v1.p.rapidapi.com/v1/preview/single/{is_id}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "landr-mastering-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def preview_gettrackmasterstatus(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate Limits: 100 requests per minute
		            "
    id: The unique id of the preview.
        
    """
    url = f"https://landr-mastering-v1.p.rapidapi.com/v1/preview/single/{is_id}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "landr-mastering-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def master_gettrackmasterstatus(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate Limits: 100 requests per minute
		            "
    id: The unique id of the master.
        
    """
    url = f"https://landr-mastering-v1.p.rapidapi.com/v1/master/single/{is_id}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "landr-mastering-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def master_getdownloadpath(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate Limits: 100 requests per minute
		            "
    id: The unique id of the master.
        
    """
    url = f"https://landr-mastering-v1.p.rapidapi.com/v1/master/single/{is_id}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "landr-mastering-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webhook_getwebhookconfiguration(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the webhook configuration.
		            "
    
    """
    url = f"https://landr-mastering-v1.p.rapidapi.com/v1/webhook"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "landr-mastering-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

