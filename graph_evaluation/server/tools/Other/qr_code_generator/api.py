import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr(text: str, fill: str='red', back_color: str='yellow', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generates Qr"
    
    """
    url = f"https://qr-code-generator108.p.rapidapi.com/"
    querystring = {'text': text, }
    if fill:
        querystring['fill'] = fill
    if back_color:
        querystring['back_color'] = back_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator108.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

