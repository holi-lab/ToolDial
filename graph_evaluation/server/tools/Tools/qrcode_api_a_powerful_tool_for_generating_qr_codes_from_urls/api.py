import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_code(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API endpoint is a simple GET request. The URL to encode should be passed as a parameter to the endpoint."
    
    """
    url = f"https://qrcode-api-a-powerful-tool-for-generating-qr-codes-from-urls.p.rapidapi.com/qr-code"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-api-a-powerful-tool-for-generating-qr-codes-from-urls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

