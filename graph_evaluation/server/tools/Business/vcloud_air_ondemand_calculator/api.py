import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ondemand_os(authkey: str, currency: str=None, datacenter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For fetching OS Data"
    authkey: for authentication
        currency: fetch os data for particular currency eg : USD
        datacenter: fetch os data for particular datacenter  eg : SCL
        
    """
    url = f"https://vipulrajdeep-vcloud-ondemand-calculator-v1.p.rapidapi.com/ondemand/os"
    querystring = {'authkey': authkey, }
    if currency:
        querystring['currency'] = currency
    if datacenter:
        querystring['datacenter'] = datacenter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vipulrajdeep-vcloud-ondemand-calculator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ondemand_sku(authkey: str, datacenter: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For fetching SKU Pricing Data"
    authkey: required for authencation
        datacenter: fetch sku pricing data for particular datacenter e.g. SCL
        currency: fetch sku pricing data for particular currency e.g. USD
        
    """
    url = f"https://vipulrajdeep-vcloud-ondemand-calculator-v1.p.rapidapi.com/ondemand/sku"
    querystring = {'authkey': authkey, }
    if datacenter:
        querystring['datacenter'] = datacenter
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vipulrajdeep-vcloud-ondemand-calculator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ondemand_datacenter(authkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For fetching Datacenter Data"
    authkey: for authentication
        
    """
    url = f"https://vipulrajdeep-vcloud-ondemand-calculator-v1.p.rapidapi.com/ondemand/datacenter"
    querystring = {'authkey': authkey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vipulrajdeep-vcloud-ondemand-calculator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

