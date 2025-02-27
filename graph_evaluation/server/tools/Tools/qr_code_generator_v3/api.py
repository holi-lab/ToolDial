import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_custom_qr_code(url: str, file_format: str='png', file_name: str='qrcode', frame: bool=None, frame_color: str='black', size: int=400, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- url: required
		- size: optional (default 200)
		- file_name: optional  (default qrcode)
		- format: optional  (default png)
		- frame: optional  (default false)
		- frame_color: optional  (default black)"
    
    """
    url = f"https://qr-code-generator49.p.rapidapi.com/qrcode"
    querystring = {'url': url, }
    if file_format:
        querystring['file_format'] = file_format
    if file_name:
        querystring['file_name'] = file_name
    if frame:
        querystring['frame'] = frame
    if frame_color:
        querystring['frame_color'] = frame_color
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

