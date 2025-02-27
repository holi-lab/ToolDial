import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_generator(value: str, option: str, media_path: str='https://s3.amazonaws.com/freecodecamp/relaxing-cat.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "endpoint for generating brand specific qr code"
    
    """
    url = f"https://brand-image-gif-qr-code-generator2.p.rapidapi.com/qr-code"
    querystring = {'value': value, 'option': option, }
    if media_path:
        querystring['media_path'] = media_path
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brand-image-gif-qr-code-generator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

