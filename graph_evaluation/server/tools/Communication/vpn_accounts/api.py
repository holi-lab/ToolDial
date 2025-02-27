import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ovpn_from_ip(ipaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the <a href='https://allweneed12.blogspot.com/2022/12/what-is-ovpn-file.html'>.ovpn</a> configuration file from the IP Address."
    
    """
    url = f"https://vpn-accounts.p.rapidapi.com/getOvpnForIp/{ipaddress}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vpn-accounts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_vpn_accounts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches all the VPN Accounts for all the Protocols.
		Use the URL field in the OpenVPN list to get the [.ovpn]('https://allweneed12.blogspot.com/2022/12/what-is-ovpn-file.html') file."
    
    """
    url = f"https://vpn-accounts.p.rapidapi.com/getVpnAccounts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vpn-accounts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

