import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_image(hash: str, returntype: str='image', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve generated image with given `hash`."
    
    """
    url = f"https://arimagesynthesizer.p.rapidapi.com/get"
    querystring = {'hash': hash, }
    if returntype:
        querystring['returnType'] = returntype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arimagesynthesizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def my_images(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all generated images' information in a JSON response. Images can be accessed at `/get` endpoint with the given hash."
    
    """
    url = f"https://arimagesynthesizer.p.rapidapi.com/my_images"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arimagesynthesizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def my_images_by_id(uniqueid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Similar to `/my_images` endpoint, returns generated images' information in a JSON response filtered by given `id`."
    
    """
    url = f"https://arimagesynthesizer.p.rapidapi.com/my_images_by_id"
    querystring = {'uniqueID': uniqueid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "arimagesynthesizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

