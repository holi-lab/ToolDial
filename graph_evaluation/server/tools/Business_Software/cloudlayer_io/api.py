import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_url_to_image(url: str, timeout: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a GET request to convert url to image, accepts simple options as query strings in URL (use POST endpoint for advanced options)."
    timeout: milliseconds (default 30000)
        
    """
    url = f"https://cloudlayer-io.p.rapidapi.com/v1/url/image"
    querystring = {'url': url, }
    if timeout:
        querystring['timeout'] = timeout
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cloudlayer-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

