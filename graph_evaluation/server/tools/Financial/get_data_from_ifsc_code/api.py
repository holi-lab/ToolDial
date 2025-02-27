import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ifsc_code_validator(ifsc_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate IFSC Codes of the Beneficiary Bank Branches of all India banks before making any payment transfer"
    
    """
    url = f"https://get-data-from-ifsc-code.p.rapidapi.com/GetIFSCData"
    querystring = {'IFSC_Code': ifsc_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-data-from-ifsc-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

