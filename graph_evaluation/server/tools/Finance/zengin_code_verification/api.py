import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zengin_code_verification(zengincode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Japan  Banks and Branch information by Zengin Code. Zengin Codes are Bank & Branch Codes issued to Japan Banks Branches for processing Domestic  Payment Transfers & International Payment Receipts."
    
    """
    url = f"https://zengin-code-verification.p.rapidapi.com/"
    querystring = {'zengincode': zengincode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zengin-code-verification.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

