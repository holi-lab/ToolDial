import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_generator(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endppint https://qrcodegenerator-47jz.onrender.com/qrcode/download/{text} will take input from user , you can replace {text} with anything and it will generate response as QR code Image."
    
    """
    url = f"https://qr-code-api-generator1.p.rapidapi.com/qrcode/download/{text}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-api-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

