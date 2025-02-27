import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scanimageurl(imageurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scan image from URL and return QR text or Barcode number."
    imageurl: Image URL which you want to scan
        
    """
    url = f"https://qr-scanner-api.p.rapidapi.com/api/QR/scanimageurl"
    querystring = {'imageUrl': imageurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-scanner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scanimageurl(imageurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scan image from URL and return QR text or Barcode number."
    imageurl: Image URL which you want to scan
        
    """
    url = f"https://qr-scanner-api.p.rapidapi.com/api/QR/scanimageurl"
    querystring = {'imageUrl': imageurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-scanner-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

