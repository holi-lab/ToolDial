import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_qr_code(text: str, color: str='000000', logo: str='https://png.pngtree.com/png-vector/20221018/ourmid/pngtree-instagram-icon-png-image_6315974.png', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a QR code with a logo (optional) in the center of the code. The logo can be provided in the form of a URL. The color of the QR code can be customized by providing a hexadecimal color code.  The response is in PNG format."
    
    """
    url = f"https://qrlogo.p.rapidapi.com/qr-code"
    querystring = {'text': text, }
    if color:
        querystring['color'] = color
    if logo:
        querystring['logo'] = logo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrlogo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

