import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_charades_word(difficulty: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random charades word with the specified difficulty. If not difficulty is supplied, a random difficulty will be used."
    
    """
    url = f"https://pictionary-charades-word-generator.p.rapidapi.com/charades"
    querystring = {}
    if difficulty:
        querystring['difficulty'] = difficulty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pictionary-charades-word-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pictionary_word(difficulty: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random pictionary word with the specified difficulty. If not difficulty is supplied, a random difficulty will be used."
    
    """
    url = f"https://pictionary-charades-word-generator.p.rapidapi.com/pictionary"
    querystring = {}
    if difficulty:
        querystring['difficulty'] = difficulty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pictionary-charades-word-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

