import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_business_information(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape Phone Email Social Media Account Company Address"
    
    """
    url = f"https://scrape-phone-email-social-media-account-company-address.p.rapidapi.com/email-social-media"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrape-phone-email-social-media-account-company-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

