import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_short_url(url: str, format: str='plain', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a short url that will automatically redirect users to your target website.
		
		Format types:
		- clickable (Provided as embeddable HTML link)
		- json (Provided as json)
		- xml (Provided as XML)
		- plain (Provided in the body, with no extra formatting)"
    
    """
    url = f"https://shortly-url-shortener.p.rapidapi.com/"
    querystring = {'url': url, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shortly-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

