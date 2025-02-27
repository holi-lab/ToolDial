import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stays_auto_complete(location: str, language_code: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto complete"
    language_code: `code `item from `languages ` endpoint

Default: en-us
        
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/auto-complete"
    querystring = {'location': location, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_properties_list_by_url(url: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by url"
    
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/properties/list-by-url"
    querystring = {'url': url, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_properties_detail_photos(language_code: str='en-us', id_detail: str='cr/la-buena-vida-cabinas', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail photos"
    language_code: `code `item from `languages `endpoint

Default: en-us
        id_detail: `idDetail `from `stays/properties/list `endpoint
        
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/properties/detail/photos"
    querystring = {}
    if language_code:
        querystring['language_code'] = language_code
    if id_detail:
        querystring['id_detail'] = id_detail
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_properties_list(location: str, rooms: int=None, filter_by: str=None, min_bathroom: int=None, min_bedroom: int=None, children_age: int=None, adults: int=None, children: int=None, currency_code: str='USD', sort_by: str=None, page: int=None, language_code: str='en-us', checkin_date: str=None, checkout_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list"
    filter_by: After calling this api, the response will include item filters. 
Get the urlId from filters[filter index] -> options[option index] -> urlId to set the value for this field.

**If there are multiple filters, they are separated by semicolons (;)**

Ex: class=1; distance=3220
        children_age: children_age must be numbers separated by commas
Ex: 1,2,4
        currency_code: `code `item from `currencies `endpoint

Default: USD
        language_code: `code `item from `languages `endpoint

Default: en-us
        checkin_date: Format: YYYY-MM-DD
Ex: 2023-07-01
        checkout_date: Format: YYYY-MM-DD
Ex: 2023-07-31
        
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/properties/list"
    querystring = {'location': location, }
    if rooms:
        querystring['rooms'] = rooms
    if filter_by:
        querystring['filter_by'] = filter_by
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if children_age:
        querystring['children_age'] = children_age
    if adults:
        querystring['adults'] = adults
    if children:
        querystring['children'] = children
    if currency_code:
        querystring['currency_code'] = currency_code
    if sort_by:
        querystring['sort_by'] = sort_by
    if page:
        querystring['page'] = page
    if language_code:
        querystring['language_code'] = language_code
    if checkin_date:
        querystring['checkin_date'] = checkin_date
    if checkout_date:
        querystring['checkout_date'] = checkout_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_properties_detail(id_detail: str, language_code: str='en-us', currency_code: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail"
    id_detail: `idDetail ` from `stays/properties/list` endpoint
        language_code: `code` item from` languages` endpoint

Default: en-us
        currency_code: `code `item from `currencies `endpoint

Default: USD
        
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/properties/detail"
    querystring = {'id_detail': id_detail, }
    if language_code:
        querystring['language_code'] = language_code
    if currency_code:
        querystring['currency_code'] = currency_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_properties_detail_policies(id_detail: str, language_code: str='en-us', currency_code: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail policies"
    id_detail: `idDetail` from `stays/properties/list` endpoint
        language_code: `code` item from `languages` endpoint

Default: en-us
        currency_code: `code` item from `currencies` endpoint

Default: USD
        
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/properties/detail/policies"
    querystring = {'id_detail': id_detail, }
    if language_code:
        querystring['language_code'] = language_code
    if currency_code:
        querystring['currency_code'] = currency_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_properties_detail_description(id_detail: str, language_code: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail description"
    id_detail: `idDetail` from `stays/properties/list` endpoint
        language_code: `code` item from `languages` endpoint

Default: en-us
        
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/properties/detail/description"
    querystring = {'id_detail': id_detail, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_properties_detail_facilities(id_detail: str, language_code: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail facilities"
    id_detail: `idDetail`  from `stays/properties/list` endpoint
        language_code: `code` item from `languages` endpoint

Default: en-us
        
    """
    url = f"https://booking-com13.p.rapidapi.com/stays/properties/detail/facilities"
    querystring = {'id_detail': id_detail, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(language_code: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currencies"
    language_code: `code ` item from `languages `endpoint
Default: en-us
        
    """
    url = f"https://booking-com13.p.rapidapi.com/currencies"
    querystring = {}
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get languages"
    
    """
    url = f"https://booking-com13.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

