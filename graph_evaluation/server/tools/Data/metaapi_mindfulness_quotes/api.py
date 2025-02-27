import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmindfulnessquote(author: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random quote about mindfulness from a list of 100 quotes."
    
    """
    url = f"https://metaapi-mindfulness-quotes.p.rapidapi.com/v1/mindfulness"
    querystring = {}
    if author:
        querystring['author'] = author
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metaapi-mindfulness-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

