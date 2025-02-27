import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_link(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This function is intended for generating a QR code based on the inputted URL. It was created for a web application where a user can input a URL and receive a QR code that can be scanned with a mobile device. 
		
		To use the function, a GET request should be sent to its address, and the parameter "url" should be included. If the parameter is absent, an error in JSON format will be returned. 
		
		Then the function creates a QR code using the qrcode library. The code is set to a certain size and colors, and then saved in PNG format in a BytesIO buffer. 
		
		Next, the function creates a link to the QR code and encodes the image to a base64 string, which is also included in the JSON response along with the link. 
		
		The final response in JSON format contains a link to the QR code and an encoded image as a string. It can be used in the application to display the QR code and scan it with a mobile device."
    
    """
    url = f"https://qr-code78.p.rapidapi.com/generate_qr"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code78.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def qr_code_image(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API provides an endpoint /generate_qr, which accepts a GET request with a url parameter. This parameter should contain the URL address that you want to encode into a QR code.
		
		As a result, users who send a GET request to the /generate_qr endpoint with a URL address will receive a PNG format QR code image in response."
    
    """
    url = f"https://qr-code78.p.rapidapi.com/qrcode.png"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code78.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

