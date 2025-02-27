import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def free_ofac_api(street_address: str='79 ul. Sadovnicheskaya', business_name: str='SME Bank', first_name: str='Gener', last_name: str='Garcia Molina', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a simple TRUE (target found) or FALSE (target not found) value n response to an OFAC scan request."
    street_address: Street Address of individual or business entity target
        business_name: Name of Business entity target.
        first_name: First Name(s) of individual target.
        last_name: Last Name(s) of individual target.
        
    """
    url = f"https://free-ofac-scan.p.rapidapi.com/"
    querystring = {}
    if street_address:
        querystring['street_address'] = street_address
    if business_name:
        querystring['business_name'] = business_name
    if first_name:
        querystring['first_name'] = first_name
    if last_name:
        querystring['last_name'] = last_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-ofac-scan.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

