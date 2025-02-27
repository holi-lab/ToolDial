import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_seo_analysis(domain: str, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get popular SEO metrics for a given domain name"
    country: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://domain-seo-analysis.p.rapidapi.com/domain-seo-analysis/"
    querystring = {'domain': domain, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-seo-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def serp_analysis(query: str, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SERP Analysis aka keyword overview. Discover thousands of great keyword ideas, analyze their ranking difficulty and calculate their traffic potential."
    country: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://domain-seo-analysis.p.rapidapi.com/serp-analysis/"
    querystring = {'query': query, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-seo-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

