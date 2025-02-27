import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cities(page: int=None, count: int=None, lang: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of cities"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/cities"
    querystring = {}
    if page:
        querystring['page'] = page
    if count:
        querystring['count'] = count
    if lang:
        querystring['lang'] = lang
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def divisionsbycountry(country_id: int, filter: str=None, page: int=None, count: int=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of divisions"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/countries/divisions/{country_id}"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if page:
        querystring['page'] = page
    if count:
        querystring['count'] = count
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language(select: int, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns language"
    select: id or iso_language_name or native_name or iso2 or iso3
        
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/language/{select}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def divisions(filter: str=None, count: int=None, page: int=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of divisions"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/divisions"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if count:
        querystring['count'] = count
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country(select: int, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns country"
    select: id or name or native_name or code or code_alpha3 or code_numeric
        
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/country/{select}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(filter: str=None, count: int=None, lang: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of languages"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/languages"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if count:
        querystring['count'] = count
    if lang:
        querystring['lang'] = lang
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def continents(filter: str=None, page: int=None, lang: str=None, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of continents"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/continents"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city(select: int, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns city"
    select: id or name or native_name or code or latitude or longitude or alternatenames
        
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/city/{select}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def division(select: int, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns division"
    select: id or name or native_name or code or full_name
        
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/division/{select}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countrybycontinent(continent_id: int, filter: str=None, count: int=None, lang: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of continents"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/continents/{continent_id}"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if count:
        querystring['count'] = count
    if lang:
        querystring['lang'] = lang
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(page: int=None, lang: str=None, count: int=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of countries"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/countries"
    querystring = {}
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    if count:
        querystring['count'] = count
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def citiesbycountry(continent_id: int, count: int=None, page: int=None, lang: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of cities"
    
    """
    url = f"https://my-geo-cloud.p.rapidapi.com/api/v1/geo/countries/cities/{country_id}"
    querystring = {'continent_id': continent_id, }
    if count:
        querystring['count'] = count
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "my-geo-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

