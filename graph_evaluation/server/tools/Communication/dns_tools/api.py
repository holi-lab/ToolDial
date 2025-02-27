import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip2dns(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns the hostname of a given IP address."
    ip: The IP address you want to resolve.
        
    """
    url = f"https://jack-dns-tools.p.rapidapi.com/dnstools.php"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jack-dns-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dns2ip(dns: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It returns the IP address of a given DNS address."
    dns: The DNS address you want to resolve.
        
    """
    url = f"https://jack-dns-tools.p.rapidapi.com/dnstools.php"
    querystring = {'dns': dns, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jack-dns-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chkprt(host: str, port: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if a port is open on a remote host."
    host: The DNS or IP address of the host you want to check.
        port: The port you want to check if open.
        
    """
    url = f"https://jack-dns-tools.p.rapidapi.com/dnstools.php"
    querystring = {'host': host, 'port': port, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jack-dns-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

