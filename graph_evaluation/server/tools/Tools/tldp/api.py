import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_registrars(limit: int=25, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the Registrars
		
		> Only some Registrars are available for now, new ones will be added in the future"
    
    """
    url = f"https://tldp.p.rapidapi.com/registrar"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_registrar_by_id(is_id: int, skip: int=0, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the info for a particular Registrar"
    
    """
    url = f"https://tldp.p.rapidapi.com/registrar/{is_id}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_registrar_by_name(name: str, skip: int=0, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the info for a particular Registrar"
    
    """
    url = f"https://tldp.p.rapidapi.com/registrar/{name}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_tlds(limit: int=25, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the TLDs, Top Level Domain"
    
    """
    url = f"https://tldp.p.rapidapi.com/tld"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_registrar_prices(name: str, skip: int=0, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the Registrars
		
		> Only some Registrars are available for now, new ones will be added in the future"
    
    """
    url = f"https://tldp.p.rapidapi.com/price/registrar/{name}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_tld_prices(name: str, skip: int=0, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the Registrars
		
		> Only some Registrars are available for now, new ones will be added in the future"
    
    """
    url = f"https://tldp.p.rapidapi.com/price/tld/{name}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_prices(limit: int=25, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the Registrars
		
		> Only some Registrars are available for now, new ones will be added in the future"
    
    """
    url = f"https://tldp.p.rapidapi.com/price"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tld_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the info for a particular Registrar"
    name: Please make sure it starts with a dot ex: **.**pt
        
    """
    url = f"https://tldp.p.rapidapi.com/tld/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tld_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the info for a particular Registrar"
    
    """
    url = f"https://tldp.p.rapidapi.com/tld/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

