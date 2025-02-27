import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_random_question(type: str='dirty', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to get a random question with optionally specifying the type of the challenge."
    type: You can choose question type from: funny, dirty.
        
    """
    url = f"https://drinking1.p.rapidapi.com/questions/random"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drinking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_challange(type: str='funny', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to get a random challenge with optionally specifying the type of the challenge."
    type: You can choose the type from: funny, dirty, sport.
        
    """
    url = f"https://drinking1.p.rapidapi.com/challenges/random"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drinking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

