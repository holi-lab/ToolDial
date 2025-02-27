import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vcard_qr_code(phone: str, name: str, email: str, size: int=20, border: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates VCARD QR Code"
    
    """
    url = f"https://qr-vcard-generator.p.rapidapi.com/vcard"
    querystring = {'phone': phone, 'name': name, 'email': email, }
    if size:
        querystring['size'] = size
    if border:
        querystring['border'] = border
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-vcard-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text_qr_code_generator(text: str, border: int=3, size: int=15, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates Text/URL QR Code"
    
    """
    url = f"https://qr-vcard-generator.p.rapidapi.com/qrcode"
    querystring = {'text': text, }
    if border:
        querystring['border'] = border
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-vcard-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

