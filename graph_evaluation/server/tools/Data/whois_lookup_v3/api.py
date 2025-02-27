import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_similarity(domain1: str, domain2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Helps to check if two domains are similar.
		
		Paremeters:
		> domain1 
		>domain2"
    domain1: first domain to compare with.
        domain2: second domain to compare with.
        
    """
    url = f"https://whois-lookup5.p.rapidapi.com/checksimilarity"
    querystring = {'domain1': domain1, 'domain2': domain2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whois-lookup5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dns_lookup(domain: str, rtype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint pulls DNS data from a domain
		
		Parameters:
		> domain - The domain to search
		> rtype - The type of record to pull. Records availables: A, PTR, MX, CNAME, TXT,NS."
    domain: The domain to search
        rtype: The rtype. i.e: A, MX, TXT, CNAME,NS, PTR
        
    """
    url = f"https://whois-lookup5.p.rapidapi.com/checkdns"
    querystring = {'domain': domain, 'rtype': rtype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whois-lookup5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ns_lookup(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns IP from a domain. (NameServer)
		
		Parameters
		-> search - The domain to search"
    
    """
    url = f"https://whois-lookup5.p.rapidapi.com/getip"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whois-lookup5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_information(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint pulls the domain information, such as IP, expiration date and much more.
		
		
		Parameters:
		-> search - The domain to search, don't add https or http. i.e: google.com"
    search: The domain to search
        
    """
    url = f"https://whois-lookup5.p.rapidapi.com/whois"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whois-lookup5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

