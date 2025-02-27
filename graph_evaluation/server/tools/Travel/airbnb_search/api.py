import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def properties_get_price(checkout: str, checkin: str, is_id: str, children: int=None, infants: int=None, pets: int=None, adults: int=None, locale: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Price"
    checkout: Format: YYYY-MM-DD
Ex: 2023-10-02

**To display the price, you should input the checkin and checkout dates**
        checkin: Format: YYYY-MM-DD
Ex: 2023-10-01

**To display the price, you should input the checkin and checkout dates**
        id: id from search API: data -> homes[index] -> listing -> id
        children: Default: 0
        infants: Default: 0
        pets: Default: 0
        adults: Default: 1
        locale: id item from the Get Languages API

Default: `en-US`
        currency: id item from the Get Currency API

Default: `USD`
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/property/get-price"
    querystring = {'checkout': checkout, 'checkin': checkin, 'id': is_id, }
    if children:
        querystring['children'] = children
    if infants:
        querystring['infants'] = infants
    if pets:
        querystring['pets'] = pets
    if adults:
        querystring['adults'] = adults
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_search_by_geo(swlat: int, nelng: int, swlng: int, nelat: int, self_check_in: bool=None, host_language: str=None, top_tier_stays: str=None, type_of_place: str=None, has_superhost: bool=None, has_instant_book: bool=None, property_type: str=None, amenities: str=None, min_bed: int=None, min_bathroom: int=None, min_bedroom: int=None, min_price: int=None, checkout: str=None, category: str=None, pets: int=None, locale: str=None, page: int=None, infants: int=None, checkin: str=None, adults: int=None, max_price: int=None, children: int=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Property By GEO"
    host_language: host_language comma-separated or empty for all types:
id of **Get Host Language Filters** endpoint
OR
Response from this API: data->filters->hostLanguage-> id

Ex: 2,4,1
        top_tier_stays: top_tier_stays comma-separated or empty for all types:
id of **Get Top Tier Stays Filter** endpoint
OR
Response from this API: data->filters->topTierStays-> id

Ex: 1,2
        type_of_place: type_of_place comma-separated or empty for all types:
id of **Get Type of Place Filters** endpoint
OR
Response from this API: data->filters->typeOfPlace-> id

Ex: Entire home/apt,Private room
        property_type: 
Property type comma-separated or empty for all types:
id of **Get Property Type Filters** endpoint
OR
Response from this API: data->filters->propertyType-> id

Ex: 2,4
        amenities: amenities comma-separated or empty for all types:
id of **Get Amenities Filters** endpoint
OR
Response from this API: data->filters->amenities-> id

Ex: 4,5
        checkout: Format: YYYY-MM-DD
        category: id of **Get Category** endpoint
OR
Response from this API: data->filters->categories-> id
Ex: Tag:677
Default category is all
        locale: id item from the **Get Languages** endpoint
Default: en-US
        checkin: Format: YYYY-MM-DD
        currency: id item from the **Get Currency**  endpoint
Default: USD
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/property/search-geo"
    querystring = {'swLat': swlat, 'neLng': nelng, 'swLng': swlng, 'neLat': nelat, }
    if self_check_in:
        querystring['self_check_in'] = self_check_in
    if host_language:
        querystring['host_language'] = host_language
    if top_tier_stays:
        querystring['top_tier_stays'] = top_tier_stays
    if type_of_place:
        querystring['type_of_place'] = type_of_place
    if has_superhost:
        querystring['has_superhost'] = has_superhost
    if has_instant_book:
        querystring['has_instant_book'] = has_instant_book
    if property_type:
        querystring['property_type'] = property_type
    if amenities:
        querystring['amenities'] = amenities
    if min_bed:
        querystring['min_bed'] = min_bed
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if min_price:
        querystring['min_price'] = min_price
    if checkout:
        querystring['checkout'] = checkout
    if category:
        querystring['category'] = category
    if pets:
        querystring['pets'] = pets
    if locale:
        querystring['locale'] = locale
    if page:
        querystring['page'] = page
    if infants:
        querystring['infants'] = infants
    if checkin:
        querystring['checkin'] = checkin
    if adults:
        querystring['adults'] = adults
    if max_price:
        querystring['max_price'] = max_price
    if children:
        querystring['children'] = children
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_rating(is_id: str, currency: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property rating"
    id: `id` from search API: data -> homes[index] -> listing -> id
        currency: `id` item from the **Get Currency** API

Default: `USD`
        locale: `id` item from the **Get Languages** API

Default: `en-US`
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/property/rating"
    querystring = {'id': is_id, }
    if currency:
        querystring['currency'] = currency
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_reviews(is_id: str, page: int=None, locale: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property reviews"
    id: `id` from search API: data -> homes[index] -> listing -> id
        page: Default: 1
        locale: `id` item from the **Get Languages** API

Default: `en-US`
        currency: `id` item from the **Get Currency** API

Default: `USD`
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/property/reviews"
    querystring = {'id': is_id, }
    if page:
        querystring['page'] = page
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_detail(is_id: str, locale: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property details"
    id: id from search API: data -> homes[index] -> listing -> id
        locale: id item from the Get Languages API

Default: `en-US`
        currency: id item from the Get Currency API

Default: `USD`
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/property/detail"
    querystring = {'id': is_id, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_search(query: str, self_check_in: bool=None, host_language: str=None, type_of_place: str=None, top_tier_stays: str=None, has_superhost: bool=None, property_type: str=None, amenities: str=None, min_bed: int=None, has_instant_book: bool=None, min_bathroom: int=None, category: str=None, min_bedroom: int=None, adults: int=None, locale: str=None, page: int=None, children: int=None, infants: int=None, currency: str=None, pets: int=None, max_price: int=None, min_price: int=None, checkout: str=None, checkin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Property"
    host_language: host_language comma-separated or empty for all types:
id of **Get Host Language Filters** endpoint
OR
Response from this API: data->filters->hostLanguage-> id

Ex: 2,4,1
        type_of_place: type_of_place comma-separated or empty for all types:
id of **Get Type of Place Filters** endpoint
OR
Response from this API: data->filters->typeOfPlace-> id

Ex: Entire home/apt,Private room
        top_tier_stays: top_tier_stays comma-separated or empty for all types:
id of **Get Top Tier Stays Filter** endpoint
OR
Response from this API: data->filters->topTierStays-> id

Ex: 1,2

        property_type: Property type comma-separated or empty for all types:
id of **Get Property Type Filters** endpoint
OR
Response from this API: data->filters->propertyType-> id

Ex: 2,4

        amenities: amenities comma-separated or empty for all types:
id of **Get Amenities Filters** endpoint
OR
Response from this API: data->filters->amenities-> id

Ex: 4,5
        category: **id** of **Get Category** endpoint 
OR 
Response from this API: data->filters->categories-> id
Ex: Tag:677
Default category is all
        locale: id item from the **Get Languages** endpoint
Default: en-US
        currency: id item from the **Get Currency** endpoint
Default: USD
        checkout: Format: YYYY-MM-DD
        checkin: Format: YYYY-MM-DD
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/property/search"
    querystring = {'query': query, }
    if self_check_in:
        querystring['self_check_in'] = self_check_in
    if host_language:
        querystring['host_language'] = host_language
    if type_of_place:
        querystring['type_of_place'] = type_of_place
    if top_tier_stays:
        querystring['top_tier_stays'] = top_tier_stays
    if has_superhost:
        querystring['has_superhost'] = has_superhost
    if property_type:
        querystring['property_type'] = property_type
    if amenities:
        querystring['amenities'] = amenities
    if min_bed:
        querystring['min_bed'] = min_bed
    if has_instant_book:
        querystring['has_instant_book'] = has_instant_book
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if category:
        querystring['category'] = category
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if adults:
        querystring['adults'] = adults
    if locale:
        querystring['locale'] = locale
    if page:
        querystring['page'] = page
    if children:
        querystring['children'] = children
    if infants:
        querystring['infants'] = infants
    if currency:
        querystring['currency'] = currency
    if pets:
        querystring['pets'] = pets
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if checkout:
        querystring['checkout'] = checkout
    if checkin:
        querystring['checkin'] = checkin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_auto_complete(query: str, locale: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find location for search"
    locale: id item from the Get Languages API
        currency: id item from the Get Currency API
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/autocomplete"
    querystring = {'query': query, 'locale': locale, 'currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_type_of_place_filters(query: str, language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Type of Place Filters"
    query: Input destination
        language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/filters/type-of-place"
    querystring = {'query': query, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_host_language_filters(query: str, language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Host Language Filters"
    query: Input destination
        language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/filters/host-language"
    querystring = {'query': query, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_accessibility_filters(query: str, language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Accessibility Filters"
    query: Input destination
        language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/filters/accessibility"
    querystring = {'query': query, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amenities_filters(query: str, language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amenities Filters"
    query: Input destination
        language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/filters/amenities"
    querystring = {'query': query, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_tier_stays_filter(query: str, language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Top Tier Stays Filter"
    query: Input destination
        language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/filters/top-tier-stays"
    querystring = {'query': query, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_type_filters(query: str, language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Property Type Filters"
    query: Input destination
        language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/filters/property-type"
    querystring = {'query': query, }
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get languages"
    
    """
    url = f"https://airbnb-search.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_category(query: str='New York, NY', language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Category"
    query: Input destination
        language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/categories"
    querystring = {}
    if query:
        querystring['query'] = query
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_currency(language_code: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Currency"
    language_code: `id` from `Get Languages` endpoint

Default: en-US
        
    """
    url = f"https://airbnb-search.p.rapidapi.com/currency"
    querystring = {}
    if language_code:
        querystring['language_code'] = language_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

