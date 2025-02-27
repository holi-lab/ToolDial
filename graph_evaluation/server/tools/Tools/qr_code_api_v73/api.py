import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_generater(url: str='www.google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The QR Code image converter API provides a simple way to generate QR code images from various types of input data, such as text, URL, or vCard. The API can be integrated into different applications to allow for the generation of QR codes on-demand. The API offers customization options, such as the size, color, and error correction level of the generated QR code image, to suit the specific needs of each user. The API operates through a RESTful API, using HTTP requests and JSON or XML responses. Endpoints are provided for generating QR codes, as well as for retrieving information about the API and its capabilities. The API is designed to be scalable, reliable, and easy to use, providing a convenient solution for generating QR codes in a wide range of applications."
    
    """
    url = f"https://qr-code-api186.p.rapidapi.com/qr"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-api186.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

