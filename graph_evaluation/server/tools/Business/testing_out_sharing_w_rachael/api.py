import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def input_test(testing: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "description of the endpoint"
    
    """
    url = f"https://testing-out-sharing-w-rachael.p.rapidapi.com/{input}"
    querystring = {'testing': testing, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-out-sharing-w-rachael.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accept_charset(mediaid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "accepting character set"
    mediaid: Character Set
        
    """
    url = f"https://testing-out-sharing-w-rachael.p.rapidapi.com/api.fortellis.io/vehicle-media/cartender/model-videos/v1/video/mediaId/{mediaid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-out-sharing-w-rachael.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lastname(lastname: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Last name of customer"
    lastname: last name here
        
    """
    url = f"https://testing-out-sharing-w-rachael.p.rapidapi.com/{lastName}"
    querystring = {}
    if lastname:
        querystring['lastName'] = lastname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testing-out-sharing-w-rachael.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

