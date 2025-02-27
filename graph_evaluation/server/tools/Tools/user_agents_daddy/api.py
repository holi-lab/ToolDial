import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mobile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /mobile endpoint of the User Agent Generator API generates a user agent for a mobile device. It returns a JSON object containing the generated user agent string. This endpoint is useful when you want to simulate requests coming from mobile devices. You can use this endpoint to test your website or application's responsiveness on different mobile devices, or to ensure that your server is configured to properly handle requests from mobile devices."
    
    """
    url = f"https://user-agents-daddy.p.rapidapi.com/mobile"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "user-agents-daddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def desktop(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /desktop endpoint of the User Agent Generator API generates a user agent for a desktop device. It returns a JSON object containing the generated user agent string. This endpoint is useful when you want to simulate requests coming from desktop devices. You can use this endpoint to test your website or application's functionality on different desktop devices, or to ensure that your server is configured to properly handle requests from desktop devices."
    
    """
    url = f"https://user-agents-daddy.p.rapidapi.com/desktop"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "user-agents-daddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /rnd endpoint of the User Agent Generator API generates a random user agent. It returns a JSON object containing the generated user agent string. This endpoint is useful when you want to generate a diverse set of user agents to improve your online presence and protect your business from fraud and scraping. For example, you can use this endpoint to rotate through a pool of user agents to avoid detection."
    
    """
    url = f"https://user-agents-daddy.p.rapidapi.com/rnd"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "user-agents-daddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tablet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /tablet endpoint of the User Agent Generator API generates a user agent for a tablet device. It returns a JSON object containing the generated user agent string. This endpoint is useful when you want to simulate requests coming from tablet devices. You can use this endpoint to test your website or application's functionality on different tablet devices, or to ensure that your server is configured to properly handle requests from tablet devices."
    
    """
    url = f"https://user-agents-daddy.p.rapidapi.com/tablet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "user-agents-daddy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

