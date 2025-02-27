import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_generator_api(url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Introducing the ultimate QR code generation API! With our easy-to-use API, you can quickly and easily generate QR codes for any URL you need. Simply pass in the URL you want to convert and our API will return a high-resolution QR code image that can be scanned by any QR code scanner. Whether you need to create QR codes for your business or personal use, our API has got you covered. Try it out today and see how simple and efficient QR code generation can be!"
    
    """
    url = f"https://qr-code-api79.p.rapidapi.com/qr"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-api79.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

