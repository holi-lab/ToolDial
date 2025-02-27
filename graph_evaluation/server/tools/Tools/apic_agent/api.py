import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_details_from_user_agent(ua: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET variant of User-Agent API - this API is the simplest way to get started with APIC Agent's user-agent parsing API. It accepts the user agent string in 'ua' parameter and returns the JSON object with parsed data."
    
    """
    url = f"https://apic-agent.p.rapidapi.com/"
    querystring = {'ua': ua, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apic-agent.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

