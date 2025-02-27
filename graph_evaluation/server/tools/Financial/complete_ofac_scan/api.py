import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ofac_api(street_address: str='79 ul. Sadovnicheskaya', last_name: str='Garcia Molina', business_name: str='SME Bank', first_name: str='Gener', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scans all Office of Foreign Assets Control (OFAC) Specially Designated Nationals & Blocked Persons List (SDN), and the Consolidated Sanctions List at the U.S. Treasury Department."
    street_address: Street Address of individual or business entity target
        last_name: Last Name(s) of individual target.
        business_name: Name of Business entity target.
        first_name: First Name(s) of individual target.
        
    """
    url = f"https://complete-ofac-scan.p.rapidapi.com/"
    querystring = {}
    if street_address:
        querystring['street_address'] = street_address
    if last_name:
        querystring['last_name'] = last_name
    if business_name:
        querystring['business_name'] = business_name
    if first_name:
        querystring['first_name'] = first_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-ofac-scan.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

