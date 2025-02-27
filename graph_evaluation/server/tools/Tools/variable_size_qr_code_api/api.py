import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_image(size: int=5, url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this end point takes a 'GET' request with url / string and an size / integer as parameters and returns a QR Code image of the desired size."
    
    """
    url = f"https://variable-size-qr-code-api.p.rapidapi.com/qr"
    querystring = {}
    if size:
        querystring['size'] = size
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "variable-size-qr-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

