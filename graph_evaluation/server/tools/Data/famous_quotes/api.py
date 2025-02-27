import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def quotes_by_quotee_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a list of quotes by the person who said it. Example, enter "Julius Caesar" for {name} here : /api/v1/quotes/{name}."
    
    """
    url = f"https://famous-quotes10.p.rapidapi.com/api/v1/quotes/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "famous-quotes10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_by_category(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Quotes by category. Example,  enter "Humor" for {category}  on the endpoint : /api/v1/category/{category}"
    
    """
    url = f"https://famous-quotes10.p.rapidapi.com/api/v1/category/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "famous-quotes10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

