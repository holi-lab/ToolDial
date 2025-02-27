import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(link: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple Get endpoint to retrieve a QR code as a Jpeg file. 
		Send the parameter **link ** as a query string parameter.
		Return content-type is application/octet-stream."
    
    """
    url = f"https://simple-qr-code-generator.p.rapidapi.com/httptriggerqrcodesingle"
    querystring = {'link': link, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

