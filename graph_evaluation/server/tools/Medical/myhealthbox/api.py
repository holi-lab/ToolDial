import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_documents(nman_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of documents associated to a specific product"
    nman_code: This value is returned by a product/info call (i.e. nman_code)
        
    """
    url = f"https://myhealthbox.p.rapidapi.com/product/documents"
    querystring = {'nman_code': nman_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myhealthbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_info(product_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available information about a specific product"
    product_id: This value is returned by a call to search/fulltext  (i.e. product_id)
        
    """
    url = f"https://myhealthbox.p.rapidapi.com/product/info"
    querystring = {'product_id': product_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myhealthbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_document_url(document_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the URL of a document, the URL is valid for 24 hours."
    document_id: This value is returned by an product/documents call
        
    """
    url = f"https://myhealthbox.p.rapidapi.com/document/getUrl"
    querystring = {'document_id': document_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myhealthbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def updated_documents(sd: str, t: str=None, is_from: int=None, ed: str=None, limit: int=None, c: str='us', f: str=None, l: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search documents that have been updated after a certain date or within a date interval"
    t: Document type, one of:  
pil (Patient Information Leaflet), spc (Summary of Product Characteristics), par (Public Assessment Report), sds (Safety Data Sheet), man (User Manual), inf (Product Information), mmr (Additional Risk Minimisation Measures), hta (Health Technology Assessment Report)
        limit: Default value: 32
Max value: 100
        c: See the Supported countries section in the Developers Manual
        f: Field optional parameter, if present it can only have 1 value. Perform date comparison on this attribute (date when record was updated in the myHealthbox database) instead of the default one (date when the actual document was updated).
        l: See the Supported countries section in the Developers Manual.
        
    """
    url = f"https://myhealthbox.p.rapidapi.com/search/updatedDocuments"
    querystring = {'sd': sd, }
    if t:
        querystring['t'] = t
    if is_from:
        querystring['from'] = is_from
    if ed:
        querystring['ed'] = ed
    if limit:
        querystring['limit'] = limit
    if c:
        querystring['c'] = c
    if f:
        querystring['f'] = f
    if l:
        querystring['l'] = l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myhealthbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_search(q: str, c: str='us', limit: int=10, l: str='en', is_from: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search alerts (health alerts, products recalls, etc.) in over 60 countries. Information is only from official sources (health Ministries and Government Agencies)."
    q: Any string or set of strings
        c: See the manual for a full list of supported countries and codes.
        limit: Default value is: 32
Maximum value: 100
        l: See the manual for a full list of supported languages and codes.
        
    """
    url = f"https://myhealthbox.p.rapidapi.com/search/alerts"
    querystring = {'q': q, }
    if c:
        querystring['c'] = c
    if limit:
        querystring['limit'] = limit
    if l:
        querystring['l'] = l
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myhealthbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fulltext_search(q: str, is_from: int=0, f: str=None, l: str='en', c: str='us', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fulltext search against the whole database of products (default search is on all fields and documents)"
    q: Any string or set of strings
        f: Only one of the following values can be used: name, active_ingredient, barcode, atc_code.
        l: See the manual for a full list of supported languages and codes.
        c: See the manual for a full list of supported countries and codes.
        limit: Default value is: 32
Maximum value: 100
        
    """
    url = f"https://myhealthbox.p.rapidapi.com/search/fulltext"
    querystring = {'q': q, }
    if is_from:
        querystring['from'] = is_from
    if f:
        querystring['f'] = f
    if l:
        querystring['l'] = l
    if c:
        querystring['c'] = c
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myhealthbox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

