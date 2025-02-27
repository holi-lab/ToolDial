import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_code_from_text(text: str, width: str='500', margin: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate QR code image from text with custom width, margin"
    
    """
    url = f"https://qr-code-generator19.p.rapidapi.com/generator"
    querystring = {'text': text, }
    if width:
        querystring['width'] = width
    if margin:
        querystring['margin'] = margin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

