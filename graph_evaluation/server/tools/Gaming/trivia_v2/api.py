import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trivia(category: str=None, difficulty: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a trivia question and answer."
    
    """
    url = f"https://trivia12.p.rapidapi.com/"
    querystring = {}
    if category:
        querystring['category'] = category
    if difficulty:
        querystring['difficulty'] = difficulty
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trivia12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

