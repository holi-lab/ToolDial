import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vat_validation_api_endpoint(vat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Input**
		
		vat: VAT identification number with country prefix, e.g. IE8256796U, GB947785557 or NO926639455
		
		**Response**
		
		JSON array."
    vat: VAT identification number with country prefix, e.g. IE8256796U
        
    """
    url = f"https://validate-vat-number.p.rapidapi.com/"
    querystring = {'vat': vat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "validate-vat-number.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

