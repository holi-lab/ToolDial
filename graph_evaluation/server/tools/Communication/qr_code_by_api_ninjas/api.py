import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_qrcode(data: str, format: str, size: int=None, fg_color: str=None, bg_color: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas QR Code API endpoint. Returns a QRCode image binary specified by input parameters."
    data: data to encode in the QR code.
        format: image format to return. Must be one of the following: png, jpg, jpeg, eps, svg.
        size: size of the QR code image to generate. The output will be a square image with (size x size) dimensions.
        fg_color: foreground color of the QR code. Must be a 6-digit hex color (e.g. 00ff00 for green). Default is 000000 (black)
        bg_color: background color of the QR code. Must be a 6-digit hex color (e.g. 00ff00 for green). Default is ffffff (white)
        
    """
    url = f"https://qr-code-by-api-ninjas.p.rapidapi.com/v1/qrcode"
    querystring = {'data': data, 'format': format, }
    if size:
        querystring['size'] = size
    if fg_color:
        querystring['fg_color'] = fg_color
    if bg_color:
        querystring['bg_color'] = bg_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

