import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(properties: str, page: int=2, per_page: str='10', callback: str='foo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Codepoints. This API exposes the functionality of the main site’s search."
    properties: one or several key/value pairs of Unicode properties or int for decimal codepoint values or q for free text search.
        page: Page number for paginated results
        per_page: results per page
        callback: JSON-P return function name
        
    """
    url = f"https://codepoints.p.rapidapi.com/search"
    querystring = {'properties': properties, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property(property: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show one Property for All Codepoints"
    property: The Unicode property to render
        
    """
    url = f"https://codepoints.p.rapidapi.com/property/{property}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def glyph(codepoint: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a Sample Glyph"
    codepoint: hexadecimal codepoint
        
    """
    url = f"https://codepoints.p.rapidapi.com/glyph/{codepoint}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter(properties: str, data: str, callback: str='foo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Filter a String, e.g. , by Unicode version or only uppercase letters"
    properties: any Unicode property. Appending a "!" negates it.
        data: an UTF-8 encoded string
        callback: JSON-P return function name
        
    """
    url = f"https://codepoints.p.rapidapi.com/filter/{data}"
    querystring = {'properties': properties, }
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def name(codepoint: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a Codepoint’s Name"
    codepoint: Hexadecimal codepoint
        callback: JSON-P function name
        
    """
    url = f"https://codepoints.p.rapidapi.com/name/{codepoint}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def codepoint(codepoint: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show detailed information about a single codepoint. You can specify fields of interest with the “property” parameter: codepoint/1234?property=age,uc,lc"
    codepoint: The hex number of the codepoint
        
    """
    url = f"https://codepoints.p.rapidapi.com/codepoint/{codepoint}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def script(callback: str, script: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about a Script"
    callback: JSON-P response function name
        script: One or more ISO 15924 script codes separated by comma
        
    """
    url = f"https://codepoints.p.rapidapi.com/script/{script}"
    querystring = {'callback': callback, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block(block: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about a Unicode block"
    block: Name of the Unicode block
        callback: JSON-P function name
        
    """
    url = f"https://codepoints.p.rapidapi.com/block/{block}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transform(action: str, data: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transform a String, e.g. , to upper-case, NFC, NFD, …"
    action: the action to be applied to {data}, one of 'lower', 'upper', 'title', 'mirror', 'nfc', 'nfd', 'nfkc', 'nfkd'
        data: The string to apply the transformation
        
    """
    url = f"https://codepoints.p.rapidapi.com/transform/{action}/{data}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def plane(plane: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about a Unicode plane"
    plane: Name of the Unicode plane
        callback: JSON-P function name
        
    """
    url = f"https://codepoints.p.rapidapi.com/plane/{plane}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "codepoints.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

