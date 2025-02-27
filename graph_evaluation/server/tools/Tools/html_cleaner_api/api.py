import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def html_text_cleanup(dirtytext: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get input string (as query-parameter) for dirty-HTML and return cleaned-HTML string."
    dirtytext: Input dirty-HTML string
        
    """
    url = f"https://html-cleaner-api.p.rapidapi.com/api/htmlcleaner/cleanuptext"
    querystring = {'dirtyText': dirtytext, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "html-cleaner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def html_url_cleanup(dirtyurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get input URL (as query-parameter) for dirty-HTML and return cleaned-HTML string."
    dirtyurl: Input dirty-HTML URL
        
    """
    url = f"https://html-cleaner-api.p.rapidapi.com/api/htmlcleaner/cleanupurl"
    querystring = {'dirtyUrl': dirtyurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "html-cleaner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

