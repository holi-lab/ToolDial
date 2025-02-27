import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stars_get_image(stars: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get image links displaying the star-rate"
    stars: The star-rate between 1 and 5
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/stars/get-image"
    querystring = {'stars': stars, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stars_get_string(stars: int, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get string describing the star-rate"
    stars: The star-rate between 1 and 5
        locale: The locale code
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/stars/get-string"
    querystring = {'stars': stars, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consumers_get_web_links(is_id: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get web links to a consumer"
    id: The value of consumer->id field returned in .../business-units/get-reviews or .../consumers/detail endpoint
        locale: The locale code
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/consumers/get-web-links"
    querystring = {'id': is_id, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consumers_get_reviews(is_id: str, includereportedreviews: bool=None, perpage: int=1, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews belonging to a consumer"
    id: The value of consumer->id field returned in .../business-units/get-reviews or .../consumers/detail endpoint
        includereportedreviews: Whether or not include reports related to a review
        perpage: The number of items per response, for paging purpose
        page: The page index, for paging purpose
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/consumers/get-reviews"
    querystring = {'id': is_id, }
    if includereportedreviews:
        querystring['includeReportedReviews'] = includereportedreviews
    if perpage:
        querystring['perPage'] = perpage
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consumers_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a consumer"
    id: The value of consumer->id field returned in .../business-units/get-reviews endpoint
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/consumers/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_units_get_web_links(is_id: str, locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get web link of a business unit"
    id: The value of id field returned in .../business-units/search or .../business-units/search-by-domain endpoint
        locale: The locale code
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/business-units/get-web-links"
    querystring = {'id': is_id, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_units_get_reviews(is_id: str, perpage: int=20, includereportedreviews: bool=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews related to a business unit"
    id: The value of id field returned in .../business-units/search or .../business-units/search-by-domain endpoint
        perpage: The number of items per response, for paging purpose
        includereportedreviews: Whether or not include reports related to a review
        page: The page index, for paging purpose
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/business-units/get-reviews"
    querystring = {'id': is_id, }
    if perpage:
        querystring['perPage'] = perpage
    if includereportedreviews:
        querystring['includeReportedReviews'] = includereportedreviews
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_units_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a business unit"
    id: The value of id field returned in .../business-units/search or .../business-units/search-by-domain endpoint
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/business-units/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_units_search_by_domain(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for exact business unit by domain name"
    name: Any domain name
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/business-units/search-by-domain"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_units_search(query: str, country: str='US', page: int=1, perpage: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for business units by term or phrase"
    query: Any term or phrase that you are familiar with
        country: The country code
        page: The page index, for paging purpose
        perpage: The number of items per response, for paging purpose
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/business-units/search"
    querystring = {'query': query, }
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if perpage:
        querystring['perpage'] = perpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_search(query: str, country: str='US', locale: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for categories in which the term or phrase belong to"
    query: Any term or phrase that you are familiar with
        country: The country code
        locale: The locale code
        
    """
    url = f"https://unofficial-trust-pilot.p.rapidapi.com/categories/search"
    querystring = {'query': query, }
    if country:
        querystring['country'] = country
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-trust-pilot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

