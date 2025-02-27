import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def famous_popular_fictional_characters_arabia_arabic_descent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most Famous Popular Fictional Characters from Arabia Arabic Descent"
    
    """
    url = f"https://famous-popular-fictional-characters-arabia-arabic-descent.p.rapidapi.com/83i0qh/famouspopularfictionalcharactersarabiaarabicdescent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "famous-popular-fictional-characters-arabia-arabic-descent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

