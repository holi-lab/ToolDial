import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_image(url: str='www.google.com', mime_type: str='image/png', color: str='red', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end point takes a 'GET' request with url / string, color / string, mime_type /  string as a parameter and returns QR code image."
    
    """
    url = f"https://custom-qr-code-api.p.rapidapi.com/generate"
    querystring = {}
    if url:
        querystring['url'] = url
    if mime_type:
        querystring['mime_type'] = mime_type
    if color:
        querystring['color'] = color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-qr-code-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

