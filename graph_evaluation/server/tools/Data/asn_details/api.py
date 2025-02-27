import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_details_by_asn_number(asn_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter an ASN Number (Integer) to get all of its details such as IP ranges, Country, Organization, CIDR."
    asn_number: ASN Number to retrieve its details
        
    """
    url = f"https://asn-details.p.rapidapi.com/asn-details/{asn_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asn-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_asn_by_country(country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter country code (ISO 2 letter code such as US, GB etc) to get all the ASNs associated with that country."
    country_code: Country Code (e.g US, UK etc) to retrieve all of its ASN numbers
        
    """
    url = f"https://asn-details.p.rapidapi.com/country2asn/{country_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asn-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_asn_by_ip(ip_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter an IP Address (Both IPv4 or IPv6 Addresses allowed) to get its ASN number, Organization, Country etc."
    ip_address: IP Address to retrieve its ASN details
        
    """
    url = f"https://asn-details.p.rapidapi.com/ip2asn/{ip_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "asn-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

