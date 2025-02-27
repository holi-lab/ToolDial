import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_qr(url: str, fill_color: str='purple', back_color: str='yellow', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This api receives a link, two optional color parameters and a parameter for the type of logo display. The colors are responsible for the background of the QR code and the color of the dots."
    
    """
    url = f"https://color-qr-code-with-your-logo1.p.rapidapi.com/create_qr"
    querystring = {'url': url, }
    if fill_color:
        querystring['fill_color'] = fill_color
    if back_color:
        querystring['back_color'] = back_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "color-qr-code-with-your-logo1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

