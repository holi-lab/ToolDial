import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_matrix_2d(min: int=1000, size: str='9x9', seed: str='anything', max: int=9999, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a 2D random numbers matrix with range, seed and size."
    
    """
    url = f"https://seedy-random-number-generator.p.rapidapi.com/random/matrix"
    querystring = {}
    if min:
        querystring['min'] = min
    if size:
        querystring['size'] = size
    if seed:
        querystring['seed'] = seed
    if max:
        querystring['max'] = max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seedy-random-number-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_list(min: int=1000, size: int=9, max: int=9999, seed: str='anything', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a list or array of random numbers with range, seed and size."
    
    """
    url = f"https://seedy-random-number-generator.p.rapidapi.com/random/list"
    querystring = {}
    if min:
        querystring['min'] = min
    if size:
        querystring['size'] = size
    if max:
        querystring['max'] = max
    if seed:
        querystring['seed'] = seed
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seedy-random-number-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_number(max: int=9999, min: int=1111, seed: str='anything', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a single random number with range and seed."
    
    """
    url = f"https://seedy-random-number-generator.p.rapidapi.com/random"
    querystring = {}
    if max:
        querystring['max'] = max
    if min:
        querystring['min'] = min
    if seed:
        querystring['seed'] = seed
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seedy-random-number-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

