import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def upc_ean_lookup(digits: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UPC / EAN BARCODE LOOKUP with Image"
    digits: Enter Either UPC or EAN
        
    """
    url = f"https://upc-ean-lookup-with-image.p.rapidapi.com/barcode/{digits}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upc-ean-lookup-with-image.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

