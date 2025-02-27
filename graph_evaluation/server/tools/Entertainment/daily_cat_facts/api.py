import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_some_random_facts(animal_type: str=None, amount: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get some random facts ."
    animal_type: Type of animal the fact will describe . Default : \"cat\"
        amount: Number of Facts to retrieve. If set to one, response will be a fact object. If many, response will be an array of Facts . 
Default : 1.
Limit : 500.
        
    """
    url = f"https://daily-cat-facts.p.rapidapi.com/facts/random"
    querystring = {}
    if animal_type:
        querystring['animal_type'] = animal_type
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-cat-facts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

