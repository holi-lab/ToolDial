import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_image(hashsize: str, hashcode: str, imagename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "after you request an image you can get it from this request
		
		`https://qr-code-for-saudi-arabia-zakat-zatca1.p.rapidapi.com/qrcode/{hashSize}/{hashCode}/{imageName}?rapidapi-key={rapidapi-key}`"
    
    """
    url = f"https://qr-code-for-saudi-arabia-zakat-zatca1.p.rapidapi.com/qrcode/{hashsize}/{hashcode}/{imagename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-for-saudi-arabia-zakat-zatca1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

