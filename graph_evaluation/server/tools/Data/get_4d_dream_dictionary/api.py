import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_dream_number(keyword: str, digit: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return relevant 4/3 digits lucky draw number based on your dream keyword in english or chinese."
    keyword: Keyword of dream to search for.
        digit: Number digit of lucky draw to search.
        language: Language of keyword.
**en** for English, **zh** for Chinese (simplified)
        
    """
    url = f"https://4d-dream-dictionary.p.rapidapi.com/get_dream_number/{digit}/{language}/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4d-dream-dictionary.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

