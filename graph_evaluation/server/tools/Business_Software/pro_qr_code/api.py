import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_code(content: str, recovery: str='M', disable_border: bool=None, bgcolor: str='ffffff', fgcolor: str='000000', size: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a QR code image with specified content and size."
    content: Text content that should be encoded in QR code.
        recovery: Recovery level. Possible values:
- `L`: 7% error recovery
- `M`: 15% error recovery (default)
- `Q`: 25% error recovery
- `H`: 30% error recovery
        disable_border: Disable the QR Code border. Default is false (border is enabled).
        bgcolor: Background color in HEX format. Default is white (`ffffff`).
        fgcolor: Foreground color in HEX format. Default is black (`000000`).
        size: Size of the QR image in pixels. Should be less than or equal 5000. Default is 500.
        
    """
    url = f"https://pro-qr-code.p.rapidapi.com/qr"
    querystring = {'content': content, }
    if recovery:
        querystring['recovery'] = recovery
    if disable_border:
        querystring['disable_border'] = disable_border
    if bgcolor:
        querystring['bgcolor'] = bgcolor
    if fgcolor:
        querystring['fgcolor'] = fgcolor
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pro-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

