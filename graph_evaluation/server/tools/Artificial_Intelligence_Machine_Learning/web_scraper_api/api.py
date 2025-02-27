import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def request(url: str, render_js: bool=None, body: str=None, headers: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a request to any URL"
    
    """
    url = f"https://web-scraper-api.p.rapidapi.com/v1/request"
    querystring = {'url': url, }
    if render_js:
        querystring['render_js'] = render_js
    if body:
        querystring['body'] = body
    if headers:
        querystring['headers'] = headers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-scraper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

