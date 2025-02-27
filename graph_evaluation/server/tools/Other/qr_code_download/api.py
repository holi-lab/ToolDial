import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_qr_code(data: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generates a QR code image when a GET request is made to this endpoint with a 'data' parameter that contains the data string to be encoded in the QR code.
		In the URL bar of browser, add the data query parameter to specify the data you want to encode in the QR code. For example, to encode the URL https://www.example.com, the URL would look like http://localhost:5000/generate_qr?data=https://www.example.com."
    
    """
    url = f"https://qr-code-download1.p.rapidapi.com/generate_qr"
    querystring = {'data': data, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-download1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

