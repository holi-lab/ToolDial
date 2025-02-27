import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_qcode(url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a GET request to the API endpoint "qcode" by passing in the URL you want to encode in the QR code as a parameter.
		
		You can also pass in additional parameters such as the size of the QR code, the error correction level and the file format you want to receive.
		
		The API will return a QR code image in the format you specified (PNG, JPG, etc.)
		
		You can download the QR code image and use it in your marketing materials, business cards or anywhere else you want to promote your product or service.
		
		You can also use the API to track the number of scans and view statistics.
		
		You can also use the API to generate and download a batch of QR codes with different urls.
		
		If you encounter any issues or have questions, feel free to reach out to our support team for assistance."
    
    """
    url = f"https://qr-code-api127.p.rapidapi.com/qcode"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-api127.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

