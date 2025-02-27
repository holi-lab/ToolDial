import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify_upc_checksum(upc: str, op: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify the checksum inside a UPC barcode"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'upc': upc, 'op': op, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upc_to_ean(op: str, upc: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a UPC code into an EAN code"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'op': op, 'upc': upc, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isbn_to_gtin(op: str, format: str='json', isbn: str='3000000119', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an ISBN number (ISBN10) into a GTIN identifier (GTIN14)"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'op': op, }
    if format:
        querystring['format'] = format
    if isbn:
        querystring['isbn'] = isbn
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gtin_to_ean(op: str, gtin: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a GTIN identifier into an EAN barcode (watch VALID flag in output if conversion is possible)"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'op': op, 'gtin': gtin, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ean_to_gtin(ean: str, op: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an EAN code into a GTIN identifier (GTIN14)"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'ean': ean, 'op': op, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isbn_to_ean(isbn: str, op: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an ISBN number (ISBN10) into an EAN code"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'isbn': isbn, 'op': op, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ean_to_isbn(ean: str, op: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an EAN code into an ISBN number (ISBN10) (check the VALID flag in the output if conversion is possible)"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'ean': ean, 'op': op, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ean_to_upc(op: str, ean: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an EAN code into a UPC code (watch VALID flag in output if conversion is possible)"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'op': op, 'ean': ean, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_ean_checksum(op: str, ean: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify the checksum inside an EAN barcode"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {'op': op, 'ean': ean, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_isbn_checksum(isbn: str='3000000119', format: str='json', op: str='verify-isbn-checksum', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify the checksum inside an ISBN number"
    
    """
    url = f"https://barcode-verification-and-conversion.p.rapidapi.com/barcode"
    querystring = {}
    if isbn:
        querystring['isbn'] = isbn
    if format:
        querystring['format'] = format
    if op:
        querystring['op'] = op
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-verification-and-conversion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

