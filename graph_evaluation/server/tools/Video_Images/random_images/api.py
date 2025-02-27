import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_single_random_image(quantum: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the metadata for a single random picture from the open images dataset, along with its human-verified labels."
    
    """
    url = f"https://random-images1.p.rapidapi.com/random"
    querystring = {}
    if quantum:
        querystring['quantum'] = quantum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-images1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_multiple_random_images(n: int, quantum: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the metadata for a multiple random pictures from the open images dataset, along with their human-verified labels."
    
    """
    url = f"https://random-images1.p.rapidapi.com/random/multiple"
    querystring = {'n': n, }
    if quantum:
        querystring['quantum'] = quantum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-images1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

