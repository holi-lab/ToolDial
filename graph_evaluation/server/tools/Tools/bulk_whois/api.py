import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_whois_batch(batch_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get WHOIS batch."
    
    """
    url = f"https://pointsdb-bulk-whois-v1.p.rapidapi.com/batch/{batch_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointsdb-bulk-whois-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulk_whois(ip: str=None, format: str='split', domain: str=None, domains: str='foo.com,example.org,google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "WHOIS query"
    ip: IP address. Will get resolved to domain.
        format: json: rich, structured json, raw: raw data, split: formatted lines
        domain: Domain name
        domains: Coma separated domain names
        
    """
    url = f"https://pointsdb-bulk-whois-v1.p.rapidapi.com/whois"
    querystring = {}
    if ip:
        querystring['ip'] = ip
    if format:
        querystring['format'] = format
    if domain:
        querystring['domain'] = domain
    if domains:
        querystring['domains'] = domains
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointsdb-bulk-whois-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_your_whois_batches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of your batches."
    
    """
    url = f"https://pointsdb-bulk-whois-v1.p.rapidapi.com/batch"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointsdb-bulk-whois-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

