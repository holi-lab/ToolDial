import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_enpoint(back_color: str, url: str, fil_color: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate QR code with URL and change colors!"
    
    """
    url = f"https://qr-codes-custom-colors.p.rapidapi.com/generate_qr"
    querystring = {'back_color': back_color, 'url': url, 'fil_color': fil_color, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-codes-custom-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

