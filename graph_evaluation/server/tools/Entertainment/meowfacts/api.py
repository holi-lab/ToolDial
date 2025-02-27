import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def defaultroot(count: int=None, factid: str=None, lang: str='eng', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By passing in the appropriate options, you can recieve a specific or more than one fact.
		"
    count: number of facts to return
        factid: pass an optional id to call a specific fact
        lang: requested locale to retrieve the cat fact in.
        
    """
    url = f"https://meowfacts.p.rapidapi.com/"
    querystring = {}
    if count:
        querystring['count'] = count
    if factid:
        querystring['factID'] = factid
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meowfacts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will list all languages available to the root endpoint.   
		"
    
    """
    url = f"https://meowfacts.p.rapidapi.com/options"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meowfacts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def healthcheck(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The health check endpoint will return a status of 200 if the api is up and ready to recieve connections. It will tell the uptime, and total requests served since last restart. It also has a field for version which corresponds to the versioned release from the github repo.  
		"
    
    """
    url = f"https://meowfacts.p.rapidapi.com/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meowfacts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

