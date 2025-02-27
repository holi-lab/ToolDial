import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crop(width: str, url: str, height: str, fit: str='contain', position: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint would crop image"
    
    """
    url = f"https://image-cropper1.p.rapidapi.com/crop"
    querystring = {'width': width, 'url': url, 'height': height, }
    if fit:
        querystring['fit'] = fit
    if position:
        querystring['position'] = position
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-cropper1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

