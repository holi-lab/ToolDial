import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def argentina_cbu_validator(cbu: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate Argentina  Clave Bancaria Única (CBU) for Bank & Branch Codes & Bank Account Numbers by Check Digits methods required for CBU Validation. Reduce wrong payments errors by using the API."
    
    """
    url = f"https://argentina-cbu-validation.p.rapidapi.com/"
    querystring = {'cbu': cbu, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "argentina-cbu-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

