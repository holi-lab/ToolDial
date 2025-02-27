import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_profanityfilter(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Profanity Filter API endpoint. Returns the censored version (bad words replaced with asterisks) of any given text and whether the text contains profanity."
    text: Input text. Maximum 1000 characters.
        
    """
    url = f"https://profanity-filter-by-api-ninjas.p.rapidapi.com/v1/profanityfilter"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "profanity-filter-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

