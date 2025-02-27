import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_v1_whois(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Entire WHOIS text, if available, AND the expiration date (YYYY-M-D) parsed from that text. Some TLDs (registries) WHOIS records are more useful than others. https://besta.domains/api"
    domain: domain name + TLD (no spaces) (ex: example.com) Documentation at https://besta.domains/api
        
    """
    url = f"https://domain-search-tools.p.rapidapi.com/v1/whois"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-search-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

