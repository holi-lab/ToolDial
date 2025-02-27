import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract(url: str, html: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extracts data from Linkedin URL (profile or company page)."
    html: Whether to output raw HTML in .raw property. Disabled by default.
        
    """
    url = f"https://linkedin-profiles1.p.rapidapi.com/extract"
    querystring = {'url': url, }
    if html:
        querystring['html'] = html
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-profiles1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search company or profile"
    type: Allowed values: person, company
        
    """
    url = f"https://linkedin-profiles1.p.rapidapi.com/search"
    querystring = {'query': query, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-profiles1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

