import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qrurl(url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint for this API is "/qr". This is the endpoint that the API listens for GET requests on. When a GET request is made to this endpoint with a "url" parameter in the query string, the API generates a QR code image of the provided URL and sends it back as the API response.
		
		The "url" parameter in the query string is the key data that this API needs to generate the QR code image. It is a required parameter and the API will return an error message if it's missing.
		
		To test this API, you would use a curl command such as curl "http://localhost:5000/qr?url=<your_url>" to send a GET request to the "/qr" endpoint with the desired URL as the "url" parameter. The API will then generate a QR code image of that URL and return it as the API response in the form of a PNG image."
    
    """
    url = f"https://qr26.p.rapidapi.com/qr"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr26.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

