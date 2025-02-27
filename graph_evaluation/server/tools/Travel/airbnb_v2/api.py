import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_property_checkout_price(checkout: str, checkin: str, propertyid: int, adults: int=None, pets: int=None, children: int=None, currency: str='USD', languageid: str=None, infants: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will return the checkout cost of the Property"
    checkout: Check-out date
        checkin: Check-in date
        propertyid: Required filed **propertyId** can be got from search property api as **id** parameter.
        adults: Number of adult guests (13 years and over). Default is set to 1.
        pets: Number of pets
        children: Number of children (2-12 years)
        currency: Default currency is **USD**. To get other available currency please call **Get Currency** API
        languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        infants: Number of infants (under 2 years)
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getPropertyCheckoutPrice"
    querystring = {'checkOut': checkout, 'checkIn': checkin, 'propertyId': propertyid, }
    if adults:
        querystring['adults'] = adults
    if pets:
        querystring['pets'] = pets
    if children:
        querystring['children'] = children
    if currency:
        querystring['currency'] = currency
    if languageid:
        querystring['languageId'] = languageid
    if infants:
        querystring['infants'] = infants
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_reviews(propertyid: int, totalrecords: int=None, offset: int=None, languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    propertyid: Required filed **propertyId** can be got from search property api as **id** parameter.
        totalrecords: Total number of record per api call.**Max limit is 40**
        offset: Please pass offset value if you want to exclude number of record from top/starting.
        languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getPropertyReviews"
    querystring = {'propertyId': propertyid, }
    if totalrecords:
        querystring['totalRecords'] = totalrecords
    if offset:
        querystring['offset'] = offset
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_availability(propertyid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    propertyid: Required filed **propertyId** can be got from search property api as **id** parameter.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/checkAvailability"
    querystring = {'propertyId': propertyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_details_v2(propertyid: int, currency: str='USD', adults: int=None, checkout: str=None, infants: int=None, checkin: str=None, children: int=None, languageid: str=None, pets: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    propertyid: Required filed **propertyId** can be got from search property api as **id** parameter.
        currency: Default currency is **USD**. To get other available currency please call **Get Currency** API
        adults: Number of adult guests (13 years and over). Default is set to 1.
        checkout: Check-out date
        infants: Number of infants (under 2 years)
        checkin: Check-in date
        children: Number of children (2-12 years)
        languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        pets: Number of pets
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v2/getPropertyDetails"
    querystring = {'propertyId': propertyid, }
    if currency:
        querystring['currency'] = currency
    if adults:
        querystring['adults'] = adults
    if checkout:
        querystring['checkOut'] = checkout
    if infants:
        querystring['infants'] = infants
    if checkin:
        querystring['checkIn'] = checkin
    if children:
        querystring['children'] = children
    if languageid:
        querystring['languageId'] = languageid
    if pets:
        querystring['pets'] = pets
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_details_depreciates(propertyid: int, adults: int=None, languageid: str=None, checkout: str=None, currency: str='USD', checkin: str=None, children: int=None, infants: int=None, pets: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    propertyid: Required filed **propertyId** can be got from search property api as **id** parameter.
        adults: Number of adult guests (13 years and over). Default is set to 1.
        languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        checkout: Check-out date
        currency: Default currency is **USD**. To get other available currency please call **Get Currency** API
        checkin: Check-in date
        children: Number of children (2-12 years)
        infants: Number of infants (under 2 years)
        pets: Number of pets
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getPropertyDetails"
    querystring = {'propertyId': propertyid, }
    if adults:
        querystring['adults'] = adults
    if languageid:
        querystring['languageId'] = languageid
    if checkout:
        querystring['checkOut'] = checkout
    if currency:
        querystring['currency'] = currency
    if checkin:
        querystring['checkIn'] = checkin
    if children:
        querystring['children'] = children
    if infants:
        querystring['infants'] = infants
    if pets:
        querystring['pets'] = pets
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_property_by_place(is_id: str, offset: str=None, category: str=None, totalrecords: str='10', currency: str='USD', infants: int=None, adults: int=1, pets: int=None, checkin: str=None, minbedrooms: int=None, pricemin: int=None, pricemax: int=None, minbeds: int=None, minbathrooms: int=None, children: int=None, checkout: str=None, property_type: str=None, self_check_in: bool=None, host_languages: str=None, super_host: bool=None, instant_book: bool=None, type_of_place: str=None, amenities: str=None, languageid: str=None, top_tier_stays: str=None, display_name: str='Chicago, IL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    id: id query parameter is **required** and it can be got from **Search Destination API**.
        offset: Please pass offset value if you want to exclude number of record from top/starting.
        category: Default category is all
To get other available category please call **Get Category API**
        totalrecords: Total number of record per api call.**Max limit is 40**
        currency: Default currency is **USD**. To get other available currency please call **Get Currency API**
        infants: Number of infants (under 2 years)
        adults: Number of adult guests (13 years and over). Default is set to 1.
        pets: Number of pets
        checkin: Check-in date
        minbedrooms: Minimum Bedrooms
        pricemin: Minimum Price
        pricemax: Maximum Price
        minbeds: Minimum Beds
        minbathrooms: Minimum Bathrooms
        children: Number of children (2-12 years)
        checkout: Check-out date
        property_type: Property Type is optional and if not passed then default is all.
You are retrieve available property_type from **Get Property Type Filter**
        host_languages: Host Languages is optional and if not passed then default is all.
You are retrieve available host_languages from **Get Host Language Filters**
        type_of_place: Type of Place is optional and if not passed then default is all.
You are retrieve available type_of_place from **Get Type of Place Filter**
        amenities: Amenities is optional and if not passed then default is all.
You are retrieve available amenities from **Get Amenities Filters**
        languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        top_tier_stays: Top Tier Stays is optional and if not passed then default is all.
You are retrieve available top_tier_stays from **Get Top Tier Stays Filter**
        display_name: **display_name** query parameter is **required** and it can be got from **Search Destination API**.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/searchPropertyByPlace"
    querystring = {'id': is_id, }
    if offset:
        querystring['offset'] = offset
    if category:
        querystring['category'] = category
    if totalrecords:
        querystring['totalRecords'] = totalrecords
    if currency:
        querystring['currency'] = currency
    if infants:
        querystring['infants'] = infants
    if adults:
        querystring['adults'] = adults
    if pets:
        querystring['pets'] = pets
    if checkin:
        querystring['checkin'] = checkin
    if minbedrooms:
        querystring['minBedrooms'] = minbedrooms
    if pricemin:
        querystring['priceMin'] = pricemin
    if pricemax:
        querystring['priceMax'] = pricemax
    if minbeds:
        querystring['minBeds'] = minbeds
    if minbathrooms:
        querystring['minBathrooms'] = minbathrooms
    if children:
        querystring['children'] = children
    if checkout:
        querystring['checkout'] = checkout
    if property_type:
        querystring['property_type'] = property_type
    if self_check_in:
        querystring['self_check_in'] = self_check_in
    if host_languages:
        querystring['host_languages'] = host_languages
    if super_host:
        querystring['super_host'] = super_host
    if instant_book:
        querystring['instant_book'] = instant_book
    if type_of_place:
        querystring['type_of_place'] = type_of_place
    if amenities:
        querystring['amenities'] = amenities
    if languageid:
        querystring['languageId'] = languageid
    if top_tier_stays:
        querystring['top_tier_stays'] = top_tier_stays
    if display_name:
        querystring['display_name'] = display_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_property(currency: str='USD', totalrecords: str='10', offset: str=None, category: str='TAB_8225', infants: int=None, children: int=None, minbeds: int=None, pricemax: int=None, checkin: str=None, minbathrooms: int=None, pricemin: int=None, pets: int=None, adults: int=1, property_type: str=None, minbedrooms: int=None, checkout: str=None, host_languages: str=None, amenities: str=None, instant_book: bool=None, self_check_in: bool=None, super_host: bool=None, type_of_place: str=None, top_tier_stays: str=None, languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    currency: Default currency is **USD**. To get other available currency please call **Get Currency API**
        totalrecords: Total number of record per api call.**Max limit is 40**
        offset: Please pass offset value if you want to exclude number of record from top/starting.
        category: Default category is all
To get other available category please call **Get Category API**
        infants: Number of infants (under 2 years)
        children: Number of children (2-12 years)
        minbeds: Minimum Beds
        pricemax: Maximum Price
        checkin: Check-in date
        minbathrooms: Minimum Bathrooms
        pricemin: Minimum Price
        pets: Number of pets
        adults: Number of adult guests (13 years and over). Default is set to 1.
        property_type: Property Type is optional and if not passed then default is all.
You are retrieve available property_type from **Get Property Type Filter**
        minbedrooms: Minimum Bedrooms
        checkout: Check-out date
        host_languages: Host Languages is optional and if not passed then default is all.
You are retrieve available host_languages from **Get Host Language Filters**
        amenities: Amenities is optional and if not passed then default is all.
You are retrieve available amenities from **Get Amenities Filters**
        type_of_place: Type of Place is optional and if not passed then default is all.
You are retrieve available type_of_place from **Get Type of Place Filter**
        top_tier_stays: Top Tier Stays is optional and if not passed then default is all.
You are retrieve available top_tier_stays from **Get Top Tier Stays Filter**
        languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/searchProperty"
    querystring = {}
    if currency:
        querystring['currency'] = currency
    if totalrecords:
        querystring['totalRecords'] = totalrecords
    if offset:
        querystring['offset'] = offset
    if category:
        querystring['category'] = category
    if infants:
        querystring['infants'] = infants
    if children:
        querystring['children'] = children
    if minbeds:
        querystring['minBeds'] = minbeds
    if pricemax:
        querystring['priceMax'] = pricemax
    if checkin:
        querystring['checkin'] = checkin
    if minbathrooms:
        querystring['minBathrooms'] = minbathrooms
    if pricemin:
        querystring['priceMin'] = pricemin
    if pets:
        querystring['pets'] = pets
    if adults:
        querystring['adults'] = adults
    if property_type:
        querystring['property_type'] = property_type
    if minbedrooms:
        querystring['minBedrooms'] = minbedrooms
    if checkout:
        querystring['checkout'] = checkout
    if host_languages:
        querystring['host_languages'] = host_languages
    if amenities:
        querystring['amenities'] = amenities
    if instant_book:
        querystring['instant_book'] = instant_book
    if self_check_in:
        querystring['self_check_in'] = self_check_in
    if super_host:
        querystring['super_host'] = super_host
    if type_of_place:
        querystring['type_of_place'] = type_of_place
    if top_tier_stays:
        querystring['top_tier_stays'] = top_tier_stays
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_destination(query: str, country: str='USA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/searchDestination"
    querystring = {'query': query, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_property_by_geo_co_ordinates_v2(nelng: int, nelat: int, swlng: int, swlat: int, nextpagecursor: str=None, currency: str='USD', children: int=None, minbeds: int=None, infants: int=None, type_of_place: str=None, minbathrooms: int=None, adults: int=1, totalrecords: str=None, pets: int=None, checkin: str=None, minbedrooms: int=None, pricemin: int=None, super_host: bool=None, amenities: str=None, property_type: str=None, pricemax: int=None, checkout: str=None, top_tier_stays: str=None, host_languages: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    nelng: Longitude of the northeastern corner of the search area
        nelat: Latitude of the northeastern corner of the search area
        swlng: Longitude of the southwestern corner of the search area
        swlat: Latitude of the southwestern corner of the search area
        currency: Default currency is **USD**. To get other available currency please call **Get Currency API**
        children: Number of children (2-12 years)
        minbeds: Minimum Beds
        infants: Number of infants (under 2 years)
        type_of_place: Type of Place is optional and if not passed then default is all.
You are retrieve available type_of_place from **Get Type of Place Filter**
        minbathrooms: Minimum Bathrooms
        adults: Number of adult guests (13 years and over). Default is set to 1.
        totalrecords: Total number of record per api call.**Max limit is 40**
        pets: Number of pets
        checkin: Check-in date
        minbedrooms: Minimum Bedrooms
        pricemin: Minimum Price
        amenities: Amenities is optional and if not passed then default is all.
You are retrieve available amenities from **Get Amenities Filters**
        property_type: Property Type is optional and if not passed then default is all.
You are retrieve available property_type from **Get Property Type Filter**
        pricemax: Maximum Price
        checkout: Check-out date
        top_tier_stays: Top Tier Stays is optional and if not passed then default is all.
You are retrieve available top_tier_stays from **Get Top Tier Stays Filter**
        host_languages: Host Languages is optional and if not passed then default is all.
You are retrieve available host_languages from **Get Host Language Filters**
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v2/searchPropertyByGEO"
    querystring = {'neLng': nelng, 'neLat': nelat, 'swLng': swlng, 'swLat': swlat, }
    if nextpagecursor:
        querystring['nextPageCursor'] = nextpagecursor
    if currency:
        querystring['currency'] = currency
    if children:
        querystring['children'] = children
    if minbeds:
        querystring['minBeds'] = minbeds
    if infants:
        querystring['infants'] = infants
    if type_of_place:
        querystring['type_of_place'] = type_of_place
    if minbathrooms:
        querystring['minBathrooms'] = minbathrooms
    if adults:
        querystring['adults'] = adults
    if totalrecords:
        querystring['totalRecords'] = totalrecords
    if pets:
        querystring['pets'] = pets
    if checkin:
        querystring['checkin'] = checkin
    if minbedrooms:
        querystring['minBedrooms'] = minbedrooms
    if pricemin:
        querystring['priceMin'] = pricemin
    if super_host:
        querystring['super_host'] = super_host
    if amenities:
        querystring['amenities'] = amenities
    if property_type:
        querystring['property_type'] = property_type
    if pricemax:
        querystring['priceMax'] = pricemax
    if checkout:
        querystring['checkout'] = checkout
    if top_tier_stays:
        querystring['top_tier_stays'] = top_tier_stays
    if host_languages:
        querystring['host_languages'] = host_languages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_property_by_geo_co_ordinates_deprecated(nelng: int, nelat: int, swlng: int, swlat: int, pricemin: int=None, checkout: str=None, minbeds: int=None, pets: int=None, minbathrooms: int=None, checkin: str=None, amenities: str=None, instant_book: bool=None, languageid: str=None, top_tier_stays: str=None, property_type: str=None, offset: str=None, currency: str='USD', children: int=None, totalrecords: str='10', adults: int=1, infants: int=None, minbedrooms: int=None, pricemax: int=None, host_languages: str=None, type_of_place: str=None, super_host: bool=None, self_check_in: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    nelng: Longitude of the northeastern corner of the search area
        nelat: Latitude of the northeastern corner of the search area
        swlng: Longitude of the southwestern corner of the search area
        swlat: Latitude of the southwestern corner of the search area
        pricemin: Minimum Price
        checkout: Check-out date
        minbeds: Minimum Beds
        pets: Number of pets
        minbathrooms: Minimum Bathrooms
        amenities: Amenities is optional and if not passed then default is all.
You are retrieve available amenities from **Get Amenities Filters**
        languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        top_tier_stays: Top Tier Stays is optional and if not passed then default is all.
You are retrieve available top_tier_stays from **Get Top Tier Stays Filter**
        property_type: Property Type is optional and if not passed then default is all.
You are retrieve available property_type from **Get Property Type Filter**
        offset: Please pass offset value if you want to exclude number of record from top/starting.
        currency: Default currency is **USD**. To get other available currency please call **Get Currency API**
        children: Number of children (2-12 years)
        totalrecords: Total number of record per api call.**Max limit is 40**
        adults: Number of adult guests (13 years and over). Default is set to 1.
        infants: Number of infants (under 2 years)
        minbedrooms: Minimum Bedrooms
        pricemax: Maximum Price
        host_languages: Host Languages is optional and if not passed then default is all.
You are retrieve available host_languages from **Get Host Language Filters**
        type_of_place: Type of Place is optional and if not passed then default is all.
You are retrieve available type_of_place from **Get Type of Place Filter**
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/searchPropertyByGEO"
    querystring = {'neLng': nelng, 'neLat': nelat, 'swLng': swlng, 'swLat': swlat, }
    if pricemin:
        querystring['priceMin'] = pricemin
    if checkout:
        querystring['checkout'] = checkout
    if minbeds:
        querystring['minBeds'] = minbeds
    if pets:
        querystring['pets'] = pets
    if minbathrooms:
        querystring['minBathrooms'] = minbathrooms
    if checkin:
        querystring['checkin'] = checkin
    if amenities:
        querystring['amenities'] = amenities
    if instant_book:
        querystring['instant_book'] = instant_book
    if languageid:
        querystring['languageId'] = languageid
    if top_tier_stays:
        querystring['top_tier_stays'] = top_tier_stays
    if property_type:
        querystring['property_type'] = property_type
    if offset:
        querystring['offset'] = offset
    if currency:
        querystring['currency'] = currency
    if children:
        querystring['children'] = children
    if totalrecords:
        querystring['totalRecords'] = totalrecords
    if adults:
        querystring['adults'] = adults
    if infants:
        querystring['infants'] = infants
    if minbedrooms:
        querystring['minBedrooms'] = minbedrooms
    if pricemax:
        querystring['priceMax'] = pricemax
    if host_languages:
        querystring['host_languages'] = host_languages
    if type_of_place:
        querystring['type_of_place'] = type_of_place
    if super_host:
        querystring['super_host'] = super_host
    if self_check_in:
        querystring['self_check_in'] = self_check_in
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_tier_stays_filter(languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getTopTierStaysFilter"
    querystring = {}
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_type_filters(languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getPropertyType"
    querystring = {}
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_type_of_place_filters(languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getTypeOfPlace"
    querystring = {}
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_host_language_filters(languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getHostLanguage"
    querystring = {}
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_amenities_filters(languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getAmenities"
    querystring = {}
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def test_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is to test if server is up and running"
    
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_currency(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getCurrency"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_category(languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getCategory"
    querystring = {}
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getLanguages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_accessibility_filters(languageid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    languageid: Enter **LanguageId** if you want a response in the requested language. To get **LanguageId** call **getLanguages** API and pass in the Id.
        
    """
    url = f"https://airbnb19.p.rapidapi.com/api/v1/getAccessibility"
    querystring = {}
    if languageid:
        querystring['languageId'] = languageid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

