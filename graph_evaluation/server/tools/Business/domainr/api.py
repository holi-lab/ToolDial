import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v2_register(domain: str, registrar: str='namecheap.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Responds with an HTTP redirect to a supporting registrar."
    domain: Domain to register.
        registrar: The registrar's root domain.
        
    """
    url = f"https://domainr.p.rapidapi.com/v2/register"
    querystring = {'domain': domain, }
    if registrar:
        querystring['registrar'] = registrar
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domainr.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

