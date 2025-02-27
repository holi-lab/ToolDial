import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_code(data: str, margin: int=3, light: str='fff', width: int=500, quality: int=1, dark: str='000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Generates a QR code that can only be used once.
		
		 - If the QR code is not used within 3minutes it will be invalidated.
		 - Timeout can be altered with the timeout parameter (maximum lifetime is 60minutes).
		 - Fully customizable"
    data: the URL content of the QR code
        margin: margin from the QR code to the edge of the image
        light: hex color of the light parts of the QR code (the background)
        width: width of QR image in pixels
        quality: Image Quality (range from 0  - 1)
        dark: hex color of the dark parts of the QR code (the dots/squares)
        
    """
    url = f"https://qrx1.p.rapidapi.com/qr"
    querystring = {'data': data, }
    if margin:
        querystring['margin'] = margin
    if light:
        querystring['light'] = light
    if width:
        querystring['width'] = width
    if quality:
        querystring['quality'] = quality
    if dark:
        querystring['dark'] = dark
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

