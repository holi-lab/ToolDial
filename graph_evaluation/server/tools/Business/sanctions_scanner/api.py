import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sanctions_scanner(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Input**
		
		Name, surname or company name.
		e.g. Putin or Alexander LUKASHENKO or ЛУКАШЕНКО or Alfa bank.
		It is also possible to use '%' at the beginning or end of a phrase, e.g. voroncov%
		
		
		**Response**
		
		JSON array."
    search: e.g. Putin or Alexander LUKASHENKO or ЛУКАШЕНКО or Alfa bank.
It is also possible to use '%' at the beginning or end of a phrase, e.g. voroncov%
        
    """
    url = f"https://sanctions-scanner.p.rapidapi.com/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sanctions-scanner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

