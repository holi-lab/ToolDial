import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def slack(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint's response shape is designed to match the requirements for slack messages. You can easily use this api to have a POST or GET for this /slack endpoint to get the same functionality as the defaultRoot endpoint, but in a slack style response.  
		"
    
    """
    url = f"https://fortune-cookie4.p.rapidapi.com/slack"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortune-cookie4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def defaultroot(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "default get request to the random fortune endpoint returning a standard response"
    
    """
    url = f"https://fortune-cookie4.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortune-cookie4.p.rapidapi.com"
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
    url = f"https://fortune-cookie4.p.rapidapi.com/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortune-cookie4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

