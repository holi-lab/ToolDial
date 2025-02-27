import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_information(domain: str, forcerefresh: str='0', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get your domain information easily"
    forcerefresh: Our data are cached for 3 days. If you only want real time data you can add this parameter in your request: **_forceRefresh=1** to force a real time fetch but this would **cost 2 times the credit**.
        format: Choose your format : Json or Xml
        
    """
    url = f"https://zozor54-whois-lookup-v1.p.rapidapi.com/"
    querystring = {'domain': domain, }
    if forcerefresh:
        querystring['_forceRefresh'] = forcerefresh
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zozor54-whois-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_information_from_ip(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reverse Whois from Ip."
    
    """
    url = f"https://zozor54-whois-lookup-v1.p.rapidapi.com/getDomainsFromIp"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zozor54-whois-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_250_random_whois(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 250 random whois from our database.
		All these whois are from registered domain."
    
    """
    url = f"https://zozor54-whois-lookup-v1.p.rapidapi.com/getRandomWhois"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zozor54-whois-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nslookup(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "NsLookup queries the specified DNS server and retrieves the requested records that are associated with the domain name you provided. These records contain information like the domain name’s IP addresses.
		
		The following types of DNS records are especially useful:
		
		- **A**: the IPv4 address of the domain.
		- **AAAA**: the domain’s IPv6 address.
		- **CNAME**: the canonical name — allowing one domain name to map on to another. This allows more than one website to refer to a single web server.
		- **MX**: the server that handles email for the domain.
		- **NS**: one or more authoritative name server records for the domain.
		- **TXT**: a record containing information for use outside the DNS server. The content takes the form name=value. This information is used for many things including authentication schemes such as SPF and DKIM."
    domain: The domain name you want to look up.
        
    """
    url = f"https://zozor54-whois-lookup-v1.p.rapidapi.com/nslookup"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zozor54-whois-lookup-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

