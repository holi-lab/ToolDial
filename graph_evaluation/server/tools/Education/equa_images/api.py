import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def equaimage(formula: str, binary: bool=None, fontsize: str='24', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get math equations as images"
    formula: Math equation/formula
        binary: Get the image as png binary. By default base64 encoded json response is returned.
        fontsize: Font size to use
        
    """
    url = f"https://math.p.rapidapi.com/image"
    querystring = {'formula': formula, }
    if binary:
        querystring['binary'] = binary
    if fontsize:
        querystring['fontsize'] = fontsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "math.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

