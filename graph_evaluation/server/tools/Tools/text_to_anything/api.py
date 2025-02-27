import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_barcode(content: str, includetext: bool=None, scale: int=3, height: int=10, type: str='code128', gettypes: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a barcode based on the given parameters"
    type: Defaults to code128
        gettypes: Indicates whether to return all types available.
        
    """
    url = f"https://text-to-anything.p.rapidapi.com/generateBarcode"
    querystring = {'content': content, }
    if includetext:
        querystring['includetext'] = includetext
    if scale:
        querystring['scale'] = scale
    if height:
        querystring['height'] = height
    if type:
        querystring['type'] = type
    if gettypes:
        querystring['getTypes'] = gettypes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-anything.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_qrcode(content: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a QRcode based on the given parameters"
    
    """
    url = f"https://text-to-anything.p.rapidapi.com/generateQR"
    querystring = {'content': content, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-to-anything.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

