import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fast_qr_code_api(url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is a simple Flask-based API that takes a 'GET' request with a URL as a query parameter, converts it to a QR code image and returns it as the API response. The response is also set to be downloadable using the Content-Disposition header set to attachment and the filename attribute set to qrcode.png. The API uses the qrcode library to convert the URL to a QR code image. The API can be accessed by making a GET request to the endpoint '/qr-code' and passing the 'url' as a query parameter. This is a useful API for developers who need to generate QR codes for their projects and want to access the functionality via an API."
    
    """
    url = f"https://fast-qr-code-api2.p.rapidapi.com/qr-code"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-qr-code-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

