import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def whois(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform WHOIS domain lookup."
    domain: Domain for which to perform WHOIS.
        
    """
    url = f"https://domain-name-search5.p.rapidapi.com/whois"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-name-search5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_availability(domain: str, currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check domain availability, including domain validation, expiration, prices, domain quality aspects (advantages / considerations) and more data."
    domain: Domain for which to get availability info.
        currency: Set the currency for domain pricing. Specified as ISO 4217 currency code (e.g. GBP), For the full list of currency codes, see: [ISO 4217 currency codes](https://www.iban.com/currency-codes).
        
    """
    url = f"https://domain-name-search5.p.rapidapi.com/availability"
    querystring = {'domain': domain, }
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-name-search5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_search(query: str, currency: str='USD', tlds: str=None, max_price: int=None, available_only: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for domains by keyword / query. Supports all 341 TLDs available on Google Domains. Each result includes availability information, pricing, quality aspects and more data available on Google Domains."
    query: Search query / keyword.
        currency: Set the currency for domain pricing. Specified as ISO 4217 currency code (e.g. GBP), For the full list of currency codes, see: [ISO 4217 currency codes](https://www.iban.com/currency-codes).
        tlds: TLDs to include in the search results, specified as a comma (,) separated list of TLDs.

**e.g.** *com*
**e.g.** *dev,info,net*
        max_price: Return available domains up to a certain price, specified in the currency value of the `currency` parameter.
        available_only: Only return available domains.
        
    """
    url = f"https://domain-name-search5.p.rapidapi.com/search"
    querystring = {'query': query, }
    if currency:
        querystring['currency'] = currency
    if tlds:
        querystring['tlds'] = tlds
    if max_price:
        querystring['max_price'] = max_price
    if available_only:
        querystring['available_only'] = available_only
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-name-search5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

