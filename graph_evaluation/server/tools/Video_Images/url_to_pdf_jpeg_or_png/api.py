import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_url(url: str, layout: str, size: str, margin: int, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a webpage into PDF or image"
    url: URL to fetch
        layout: landscape or portrait
        size: A3 or A4
        margin: Outside margins
        format: pdf,png or jpeg
        
    """
    url = f"https://url-to-pdf-jpeg-or-png.p.rapidapi.com/api/generate"
    querystring = {'url': url, 'layout': layout, 'size': size, 'margin': margin, 'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-to-pdf-jpeg-or-png.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

