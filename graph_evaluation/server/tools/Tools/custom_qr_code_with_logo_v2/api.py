import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def simple_qr_code(link: str, color: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoin returns simple qr code with the possiblity to customize the color."
    color: Optional. Default: #000
        
    """
    url = f"https://custom-qr-code-with-logo2.p.rapidapi.com/simple-qrcode"
    querystring = {'link': link, }
    if color:
        querystring['color'] = color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-qr-code-with-logo2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

