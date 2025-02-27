import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_url_to_qr(data: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our innovative URL QR Code technology allows users to easily input a website URL and retrieve a unique QR code that can be scanned for quick access to the website. Simply enter the desired website URL and within seconds you will have a customized QR code that can be printed or shared digitally."
    
    """
    url = f"https://qrcode-express.p.rapidapi.com/qrcode"
    querystring = {'data': data, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

