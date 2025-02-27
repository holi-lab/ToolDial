import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_code(format: str, data: str, size: int, error_correction: str=None, image_margin: int=10, image_size: str=None, image_id: str='f222c8be0475292b2a23a82ff93ac496', gradient: str='024bda,8572ea', color: str=None, background_gradient: str=None, background_color: str='transparent', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a beautiful QR Codes with custom logo, colors, gradient effects and styles with ease."
    
    """
    url = f"https://webit-qr-code.p.rapidapi.com/generate"
    querystring = {'format': format, 'data': data, 'size': size, }
    if error_correction:
        querystring['error_correction'] = error_correction
    if image_margin:
        querystring['image_margin'] = image_margin
    if image_size:
        querystring['image_size'] = image_size
    if image_id:
        querystring['image_id'] = image_id
    if gradient:
        querystring['gradient'] = gradient
    if color:
        querystring['color'] = color
    if background_gradient:
        querystring['background_gradient'] = background_gradient
    if background_color:
        querystring['background_color'] = background_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

