import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def business_email_compromise_bec_api(url: str='bill@microsoft.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API protect your business from Business email compromise (BEC), 
		it takes email and returns  keys risk indicators such as : email
		validation, blacklisted, phishing, dmarc seurity,  spoofing risk, malicious activity."
    
    """
    url = f"https://business-email-compromise-bec-api.p.rapidapi.com/url/"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-email-compromise-bec-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

