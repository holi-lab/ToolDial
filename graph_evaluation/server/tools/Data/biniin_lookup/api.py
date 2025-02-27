import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bin_iin_lookup(bin: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API for Lookup, check, or search Issuer card information using card BIN.
		Input first 6 digits number of the Credit/Debit Card"
    
    """
    url = f"https://bin-iin-lookup1.p.rapidapi.com/bins/cards.php"
    querystring = {'bin': bin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bin-iin-lookup1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

