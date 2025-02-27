import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_advance_direct_image(data: str, foreground_color: str='FF2400', background_color: str='00DBFF', size: int=500, margin: int=10, label: str='My label', label_size: int=20, label_alignment: str='center', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a QR code as a direct image with additional settings. (NOTE: doesn't show correctly in RapidAPI)"
    
    """
    url = f"https://qr-code-generator20.p.rapidapi.com/generateadvanceimage"
    querystring = {'data': data, }
    if foreground_color:
        querystring['foreground_color'] = foreground_color
    if background_color:
        querystring['background_color'] = background_color
    if size:
        querystring['size'] = size
    if margin:
        querystring['margin'] = margin
    if label:
        querystring['label'] = label
    if label_size:
        querystring['label_size'] = label_size
    if label_alignment:
        querystring['label_alignment'] = label_alignment
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_basic_base64(data: str, size: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a QR code as base64 with limited settings."
    
    """
    url = f"https://qr-code-generator20.p.rapidapi.com/generatebasicbase64"
    querystring = {'data': data, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_advance_base64(data: str, background_color: str='00DBFF', foreground_color: str='FF2400', label: str='My label', margin: int=10, size: int=500, label_size: int=20, label_alignment: str='center', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a QR code as base64 with additional settings."
    
    """
    url = f"https://qr-code-generator20.p.rapidapi.com/generateadvancebase64"
    querystring = {'data': data, }
    if background_color:
        querystring['background_color'] = background_color
    if foreground_color:
        querystring['foreground_color'] = foreground_color
    if label:
        querystring['label'] = label
    if margin:
        querystring['margin'] = margin
    if size:
        querystring['size'] = size
    if label_size:
        querystring['label_size'] = label_size
    if label_alignment:
        querystring['label_alignment'] = label_alignment
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_basic_direct_image(data: str, size: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a QR code as a direct image with limited settings. (NOTE: doesn't show correctly in RapidAPI)"
    
    """
    url = f"https://qr-code-generator20.p.rapidapi.com/generatebasicimage"
    querystring = {'data': data, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator20.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

