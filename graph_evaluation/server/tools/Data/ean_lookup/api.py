import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_search(name: str, op: str, page: int=0, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search barcode database by keyword"
    name: keywords to search for
        op: operation
        page: page through output
        format: output format (json or xml)
        
    """
    url = f"https://ean-lookup.p.rapidapi.com/api"
    querystring = {'name': name, 'op': op, }
    if page:
        querystring['page'] = page
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ean-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def barcode_lookup(op: str, ean: int, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup product by barcode"
    op: operation
        ean: barcode to lookup
        format: output format (json or xml)
        
    """
    url = f"https://ean-lookup.p.rapidapi.com/api"
    querystring = {'op': op, 'ean': ean, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ean-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def barcode_prefix_search(prefix: int, op: str, page: int=0, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for barcodes starting with this prefix"
    prefix: barcode prefix (between 4 to 12 digits)
        op: operation
        page: page through output
        format: output format (json or xml)
        
    """
    url = f"https://ean-lookup.p.rapidapi.com/api"
    querystring = {'prefix': prefix, 'op': op, }
    if page:
        querystring['page'] = page
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ean-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_checksum(op: str, ean: int, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify EAN barcode checksum"
    op: operation
        ean: barcode to verify
        format: output format (json or xml)
        
    """
    url = f"https://ean-lookup.p.rapidapi.com/api"
    querystring = {'op': op, 'ean': ean, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ean-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

