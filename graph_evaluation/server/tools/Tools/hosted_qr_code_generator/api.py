import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def uuid_qr_png(uuid: str, download: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generates a qr and hosts in the cloud and gives back URL to the hosted file"
    
    """
    url = f"https://hosted-qr-code-generator.p.rapidapi.com/{uuid}/qr.png"
    querystring = {}
    if download:
        querystring['download'] = download
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hosted-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

