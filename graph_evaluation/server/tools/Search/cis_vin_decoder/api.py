import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vindecode(vin: str, passempty: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decodes the provided North American vin and provides recall information if available. We require at least the first 12 out of 17 characters in the vin to attempt a decode. The vin is not case sensitive. If passEmpty (default False) is True we will also include the empty fields in the response json."
    
    """
    url = f"https://cis-vin-decoder.p.rapidapi.com/vinDecode"
    querystring = {'vin': vin, }
    if passempty:
        querystring['passEmpty'] = passempty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cis-vin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

