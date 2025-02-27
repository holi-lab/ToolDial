import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_code(link: str='https://www.google.com/', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Add your link and generate QR code image"
    
    """
    url = f"https://generate-qr-code-with-javascript.p.rapidapi.com/qrcode"
    querystring = {}
    if link:
        querystring['link'] = link
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "generate-qr-code-with-javascript.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

