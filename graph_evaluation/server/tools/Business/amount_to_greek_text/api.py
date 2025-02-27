import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def money(m: str, f: str='lower', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts any numerical amount to greek text
		
		"f" parameter accepted values are:
		f=title
		f=sentence
		f=upper
		f=lower"
    m: money to convert
        f: Accepted values:
f=title
f=sentence
f=upper
f=lower
        
    """
    url = f"https://amount-to-greek-text.p.rapidapi.com/money"
    querystring = {'m': m, }
    if f:
        querystring['f'] = f
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amount-to-greek-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

