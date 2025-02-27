import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def custom_qr_codes(config: str, data: str, size: int=1000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can customize your QR Code Body Shape, eye shape, eyeball shape, background color, body color, eye color, eye ball color, logo, size and output format."
    config: You can define Body Shape, eye shapes, eyeball shapes.
You can define Background Color, body color,  eyes color, eyeballs color.
Define logo to displayed on the QR Code.
Define output format 
        data: Data to be encoded in the QR Code. You can encode text, url, EMAIL, PHONE, SMS, VCARD, LOCATION, FACEBOOK Profile page, TWITTER Page, YOUTUBE Page, WIFI Settings, EVENT Detail, BITCOIN Detail
For new line, send \r\n  in your text.
        size: Size from 50-2000 in pixels
        
    """
    url = f"https://dynamic-designers-qr-code.p.rapidapi.com/custom"
    querystring = {'config': config, 'data': data, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dynamic-designers-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

