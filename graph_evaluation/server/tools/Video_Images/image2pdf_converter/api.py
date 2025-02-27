import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_image_to_pdf(image_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to convert an image file to a PDF file. To use the endpoint, send a GET request to the endpoint URL with the "image_url" (only from https://i.imgur.com/) parameter and the URL of the image you wish to convert. The API will then return the converted PDF file as a download.
		Example of how to use the endpoint:
		
		GET http://api-url.com/convert_to_pdf?image_url=https://i.imgur.com/your-image-url.png"
    
    """
    url = f"https://image2pdf-converter.p.rapidapi.com/convert_to_pdf"
    querystring = {'image_url': image_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image2pdf-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

