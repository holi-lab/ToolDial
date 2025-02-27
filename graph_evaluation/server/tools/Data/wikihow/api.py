import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def steps(count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a random selection of WikiHow steps."
    count: The number of step strings to retrieve (maximum 100)
        
    """
    url = f"https://hargrimm-wikihow-v1.p.rapidapi.com/steps"
    querystring = {'count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hargrimm-wikihow-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the src URL for random WikiHow images"
    count: The number of image URLs to retrieve (maximum 100)
        
    """
    url = f"https://hargrimm-wikihow-v1.p.rapidapi.com/images"
    querystring = {'count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hargrimm-wikihow-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

