import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ifsc_code_validator(ifsccode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate IFSC Codes of the Beneficiary Bank Branches of all India banks before making any payment transfer.  Avoid payment rejections by validating IFSC Code of  payment recipient Bank Branch."
    
    """
    url = f"https://ifsc-code-validator.p.rapidapi.com/ifscapi"
    querystring = {'ifsccode': ifsccode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ifsc-code-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

