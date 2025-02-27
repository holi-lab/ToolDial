import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gs1code128(data: str, company_prefix: str, lot: str='AB123', qty: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "it gets data, company_prefix, lot and quantity and it generates a barcode. company_prefix and data is required."
    
    """
    url = f"https://gs1-code128-generator.p.rapidapi.com/gs1code128"
    querystring = {'data': data, 'company_prefix': company_prefix, }
    if lot:
        querystring['lot'] = lot
    if qty:
        querystring['qty'] = qty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gs1-code128-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

