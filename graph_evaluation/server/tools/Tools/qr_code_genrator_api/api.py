import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_image_copy(content_disposition: str='attachment; filename= qrcode.png', content_length: str='600', content_type: str='image/png', url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The QR Code Image Generator API is a powerful tool for creating custom QR codes on-the-fly. With this API, you can generate a QR code image for any URL or string parameter that you provide. Simply make a GET request to the API with your desired URL or string as a parameter, and you'll receive a PNG image of your custom QR code that can be used for a variety of purposes."
    
    """
    url = f"https://qr-code-genrator-api1.p.rapidapi.com/qr"
    querystring = {}
    if content_disposition:
        querystring['Content-Disposition'] = content_disposition
    if content_length:
        querystring['Content-Length'] = content_length
    if content_type:
        querystring['Content-Type'] = content_type
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-genrator-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

