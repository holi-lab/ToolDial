import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cloudflare_bypass(url: str, proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# important!!
		
		If you are using a proxy in whitelist mode, please add the  ip `138.201.37.238`  to the whitelist"
    
    """
    url = f"https://cloudflare-bypass2.p.rapidapi.com/"
    querystring = {'url': url, }
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cloudflare-bypass2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cloudflare_cookie_cf_clearance(url: str, proxy: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "the cf_clearance is bound by user-agent and ip.
		
		so you need to pass on your own proxy information. such as `http://xx.xx.xx.xx:8888`
		
		# important!!
		
		If you are using a proxy in whitelist mode, please add the  ip `138.201.37.238`  to the whitelist"
    
    """
    url = f"https://cloudflare-bypass2.p.rapidapi.com/cookie"
    querystring = {'url': url, 'proxy': proxy, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cloudflare-bypass2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

