import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_information(type: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data points of a given domain : A record, MX record, TXT/SPF record, SOA record, CMS Type (Generator), Certificate Issuer, Certificate Subject, Certificate Start Date, Certificate End Date, Domain creation date, Domain name (or partial), Domain expire date, Domain updated date, Javascript Libraries, Hostname (or partial), HTML Header Tags,  IP Address, Page title from domain homepage, Page meta tags (including og), Organization, Registrar, TLD, Server software (Apache, PHP, nginx etc), Server geolocation"
    
    """
    url = f"https://livescan-a-domain.p.rapidapi.com/"
    querystring = {'type': type, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescan-a-domain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

