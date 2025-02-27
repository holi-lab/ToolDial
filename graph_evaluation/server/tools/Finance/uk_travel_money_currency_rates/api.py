import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_info(action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information on the currencies that are available and the merchants that currency data is available from."
    action: Information on currencies available (CUR), merchant list (MER) or both (ALL)
        
    """
    url = f"https://touchtree-uktravelmoneycurrencyrates.p.rapidapi.com/GetInfo"
    querystring = {'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "touchtree-uktravelmoneycurrencyrates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rates(iso: str, orderedby: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rates data on a particular currency using it's ISO code. Rates are updated every 10 minutes. Some currencies may return 0 results because no merchants have rates for them."
    iso: ISO Currency Code
        orderedby: Method of ordering rates data (Descending & Ascending by value or Alpha A to Z & Z to A by merchant name):
        
    """
    url = f"https://touchtree-uktravelmoneycurrencyrates.p.rapidapi.com/GetRates"
    querystring = {'ISO': iso, 'OrderedBy': orderedby, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "touchtree-uktravelmoneycurrencyrates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

