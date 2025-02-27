import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_specific_qr_code(ref: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From this endpoint is to get the information of a specific registered qr code, related to the respective api key"
    
    """
    url = f"https://qr-code-dynamic-and-static1.p.rapidapi.com/qrcode/{ref}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-dynamic-and-static1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_qr_code_s(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From this endpoint is to list all registered qr codes, related to their api key."
    
    """
    url = f"https://qr-code-dynamic-and-static1.p.rapidapi.com/qrcode"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-dynamic-and-static1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_api_key(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From this endpoint you can generate a single api key.
		
		It will be used to link all the qr code created, and allow any changes.
		
		Important: Keep in mind that once generated, it is necessary that you store this key well, since it is not possible to recover it, and without it, you do not have access to change any of your QR Codes."
    
    """
    url = f"https://qr-code-dynamic-and-static1.p.rapidapi.com/user/generate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-dynamic-and-static1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

