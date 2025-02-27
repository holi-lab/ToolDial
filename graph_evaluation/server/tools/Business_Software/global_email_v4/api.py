import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def global_email_v4(opt: str, email: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Define Input Fields"
    opt: Express/Premium
        email: Input Email
        format: Format of Response
        
    """
    url = f"https://global-email-v4.p.rapidapi.com/v4/WEB/GlobalEmail/doGlobalEmail"
    querystring = {'opt': opt, 'email': email, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-email-v4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

