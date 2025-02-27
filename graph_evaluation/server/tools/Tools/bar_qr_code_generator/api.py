import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download(type: str, data: str, size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate and download a single barcode."
    type: The type of barcode can be one of qr_code, code_128a, code_128b, code_128c, code_25, code_25_interleaved, code_25_iata, code_39, code_93, gs1_128, ean_13, bookland, ean_8, upc_supplemental, codabar
        data: The data to be encoded
        
    """
    url = f"https://bar-qr-code-generator.p.rapidapi.com/encoded/api/download"
    querystring = {'type': type, 'data': data, }
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bar-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

