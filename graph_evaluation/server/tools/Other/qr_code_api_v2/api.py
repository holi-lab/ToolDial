import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_generator(value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our QR code generator API endpoint allows you to easily create custom QR codes by making a GET request with a url parameter. The API will respond with a downloadable PNG image of the QR code, which can be used for marketing campaigns, product launches, and events. It's fast, reliable, and fully customizable to match your brand. Try it now and see the difference it can make for your project!"
    
    """
    url = f"https://qr-code-api63.p.rapidapi.com/qr"
    querystring = {'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-api63.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

