import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_image_generator(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A QR code generator API is a tool that enables developers to generate QR codes within their applications. QR codes are two-dimensional barcodes that can be scanned using a smartphone camera and decoded using QR code reader software. The API allows developers to easily integrate QR code generation functionality into their existing applications, such as mobile apps, web apps, and software.
		
		This API can be used to generate QR codes for a variety of purposes, such as:
		
		Contact information: Generate a QR code containing your contact information, such as your phone number or email address.
		Links: Create a QR code that links to a website, YouTube video, or other online content.
		Payments: Create a QR code for making payments through a mobile wallet or payment app.
		Events: Generate a QR code for an event, such as a concert or conference, to provide attendees with all the necessary information.
		Coupons: Create a QR code for a coupon or promo code to be redeemed at a physical store or online.
		Overall, a QR code generator API is a versatile tool that can help businesses and individuals streamline their processes and improve the user experience for their customers."
    
    """
    url = f"https://qr-code-generator-api20.p.rapidapi.com/qr"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator-api20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

