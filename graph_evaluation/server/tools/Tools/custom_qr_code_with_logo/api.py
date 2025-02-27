import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_custom_qr_code_with_get(data: str, config: str='{"bodyColor": "#0277BD", "body":"mosaic"}', download: bool=None, file: str='png', size: int=600, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create custom QR Code with logo, color and design."
    data: The content of your QR code like a URL, text or any other formats that work with QR Codes. String should be URL encoded.
        config: JSON object for configuration of custom QR code design. If you add a logo file to your config make sure you uploaded the image before. JSON needs to be URL encoded.
        download: Set to true to force your browser to download your created file instead of displaying it.
        file: Set the type of your output image file. There are different formats available: png, jpg, svg (support all design options), pdf (no support for color gradients), eps (only classic QR codes).
        size: The minimum width and height of your QR code image in pixel.
        
    """
    url = f"https://qrcode-monkey.p.rapidapi.com/qr/custom"
    querystring = {'data': data, }
    if config:
        querystring['config'] = config
    if download:
        querystring['download'] = download
    if file:
        querystring['file'] = file
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-monkey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_transparent_qr_code_with_get(data: str, size: int, y: int=0, crop: bool=None, download: bool=None, image: str=None, x: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a direct link for a transparent QR code."
    data: The content of your QR code like a URL, text or any other formats that work with QR Codes. String should be URL encoded.
        size: The width and height of the QR code displayed on your image.
        y: The y position of your QR code in your image canvas.
        crop: Return only the QR code without the surrounding image when set to true.
        download: Set to true to force direct download of image file in Browser.
        image: The filename of the uploaded image or an image URL. This image is the background and canvas for your QR code.
        x: The x position of your QR code in your image canvas.
        
    """
    url = f"https://qrcode-monkey.p.rapidapi.com/qr/transparent"
    querystring = {'data': data, 'size': size, }
    if y:
        querystring['y'] = y
    if crop:
        querystring['crop'] = crop
    if download:
        querystring['download'] = download
    if image:
        querystring['image'] = image
    if x:
        querystring['x'] = x
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-monkey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

