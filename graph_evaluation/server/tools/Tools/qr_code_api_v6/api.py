import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_image_generator(url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is a QR code generation service built using Flask. It accepts a 'url' parameter in the GET request and returns a PNG image of the generated QR code. The QR code can be saved as an attachment with the filename 'qr_code.png'. The API runs on port 4000 and can be easily integrated into any application or website that needs to generate QR codes."
    
    """
    url = f"https://qr-code-api80.p.rapidapi.com/qr"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-api80.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

