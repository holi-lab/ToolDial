import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def url_to_image(url: str, element: str='.welcome-area', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "converts site (url) into image"
    element: put a css element, by default we use body
        
    """
    url = f"https://html-to-image.p.rapidapi.com/convert/url-to-image"
    querystring = {'url': url, }
    if element:
        querystring['element'] = element
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "html-to-image.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

