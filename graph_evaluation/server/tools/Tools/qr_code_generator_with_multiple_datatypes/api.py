import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getqrcode(type: str, value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will return you an URL of the QR Code image that you can easily embed on to website or download it for your other needs."
    type: Select one of the QR code formats to generate. url,text,telno,mailto,smsto
        value: Enter value as per your QR code type. Eg: url=https://www.google.com,text=Some text,telno=9142545474,smsto=8542487542
        
    """
    url = f"https://codzz-qr-cods.p.rapidapi.com/getQrcode"
    querystring = {'type': type, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codzz-qr-cods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

