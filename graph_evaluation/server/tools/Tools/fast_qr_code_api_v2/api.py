import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_generation_api(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The QR Code Generation API is a simple API that allows users to generate QR code images for any given URL.
		
		- GET /qr - Generate QR code for the provided URL
		
		Responses
		
		**Success - 200 OK**
		
		- Content-Type: image/png
		- Content-Disposition: attachment; filename=qr-code.png
		- The response is a QR code image.
		Additional headers with status code and message are also included in the response
		
		**Error - 400 Bad Request**
		
		- Content-Type:  application/json
		- Body:  `{
		  "error": "No URL provided",
		  "status_code": 400
		}`"
    
    """
    url = f"https://fast-qr-code-api1.p.rapidapi.com/qr"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-qr-code-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

