import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def property_rental_listings(state: str='TX', status: str=None, daysold: int=None, bedrooms: int=None, propertytype: str=None, latitude: int=None, zipcode: str=None, bathrooms: int=None, address: str=None, limit: int=10, radius: int=None, offset: int=None, city: str='Austin', longitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for rental listings in a geographical area or around a specific location. Returns the listed rent, date, property features, and other listing information. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    state: The two-character state abbreviation, used to search for listings in a specific state
        status: The current listing status, used to search for listings matching this criteria. Supported values are: `Active`, `Inactive`
        daysold: The maximum number of days since a property was listed on the market, with a minimum of 1
        bedrooms: The number of bedrooms, used to search for listings matching this criteria
        propertytype: The type of the property, used to search for listings matching this criteria. Supported values are: `Single Family`, `Condo`, `Townhouse`, `Manufactured`, `Duplex-Triplex`, `Apartment`
        latitude: The latitude of the search area. Use the `latitude`/`longitude` and `radius` parameters to search for listings in a specific area
        zipcode: The 5-digit zip code, used to search for listings in a specific zip code
        bathrooms: The number of bathrooms, used to search for listings matching this criteria. Supports fractions to indicate partial bathrooms
        address: The address of the property, in the format of 'Street, City, State, Zip'. Used to retrieve data for a specific property, or together with the `radius` parameter to search for listings in a specific area
        limit: The maximum number of listings to return, between 1 and 500. Defaults to 50 if not provided
        radius: The radius of the search area, in kilometers. Use in combination with the `latitude`/`longitude` or `address` parameters to search for listings in a specific area
        offset: The index of the first listing to return, used to paginate through large lists of results. Defaults to 0 if not provided
        city: The name of the city, used to search for listings in a specific city
        longitude: The longitude of the search area. Use the `latitude`/`longitude` and `radius` parameters to search for listings in a specific area
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/rentalListings"
    querystring = {}
    if state:
        querystring['state'] = state
    if status:
        querystring['status'] = status
    if daysold:
        querystring['daysOld'] = daysold
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if propertytype:
        querystring['propertyType'] = propertytype
    if latitude:
        querystring['latitude'] = latitude
    if zipcode:
        querystring['zipCode'] = zipcode
    if bathrooms:
        querystring['bathrooms'] = bathrooms
    if address:
        querystring['address'] = address
    if limit:
        querystring['limit'] = limit
    if radius:
        querystring['radius'] = radius
    if offset:
        querystring['offset'] = offset
    if city:
        querystring['city'] = city
    if longitude:
        querystring['longitude'] = longitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_sale_listings(daysold: int=None, status: str=None, propertytype: str=None, bathrooms: int=None, zipcode: str=None, state: str='TX', latitude: int=None, bedrooms: int=None, radius: int=None, limit: int=10, offset: int=None, city: str='Austin', longitude: int=None, address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for sale listings in a geographical area or around a specific location. Returns the listed price, date, property features, and other listing information. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    daysold: The maximum number of days since a property was listed on the market, with a minimum of 1
        status: The current listing status, used to search for listings matching this criteria. Supported values are: `Active`, `Inactive`
        propertytype: The type of the property, used to search for listings matching this criteria. Supported values are: `Single Family`, `Condo`, `Townhouse`, `Manufactured`, `Duplex-Triplex`, `Apartment`, `Land`
        bathrooms: The number of bathrooms, used to search for listings matching this criteria. Supports fractions to indicate partial bathrooms
        zipcode: The 5-digit zip code, used to search for listings in a specific zip code
        state: The two-character state abbreviation, used to search for listings in a specific state
        latitude: The latitude of the search area. Use the `latitude`/`longitude` and `radius` parameters to search for listings in a specific area
        bedrooms: The number of bedrooms, used to search for listings matching this criteria
        radius: The radius of the search area, in kilometers. Use in combination with the `latitude`/`longitude` or `address` parameters to search for listings in a specific area
        limit: The maximum number of listings to return, between 1 and 500. Defaults to 50 if not provided
        offset: The index of the first listing to return, used to paginate through large lists of results. Defaults to 0 if not provided
        city: The name of the city, used to search for listings in a specific city
        longitude: The longitude of the search area. Use the `latitude`/`longitude` and `radius` parameters to search for listings in a specific area
        address: The address of the property, in the format of 'Street, City, State, Zip'. Used to retrieve data for a specific property, or together with the `radius` parameter to search for listings in a specific area
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/saleListings"
    querystring = {}
    if daysold:
        querystring['daysOld'] = daysold
    if status:
        querystring['status'] = status
    if propertytype:
        querystring['propertyType'] = propertytype
    if bathrooms:
        querystring['bathrooms'] = bathrooms
    if zipcode:
        querystring['zipCode'] = zipcode
    if state:
        querystring['state'] = state
    if latitude:
        querystring['latitude'] = latitude
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if radius:
        querystring['radius'] = radius
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if city:
        querystring['city'] = city
    if longitude:
        querystring['longitude'] = longitude
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_rental_listing_by_id(propertyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single property rental listing matching the specified id, if it exists. Property ids can be obtained via the `/properties`, `/saleListings` or `/rentalListings` endpoints. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    propertyid: The id of the property listing to return
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/rentalListings/{propertyid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_sale_listing_by_id(propertyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single property sale listing matching the specified id, if it exists. Property ids can be obtained via the `/properties`, `/saleListings` or `/rentalListings` endpoints. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    propertyid: The id of the property listing to return
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/saleListings/{propertyid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_record_by_id(propertyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single property record matching the specified id, if it exists. Property ids can be obtained via the `/properties`, `/saleListings` or `/rentalListings` endpoints. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    propertyid: The id of the property record to return
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/properties/{propertyid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_records(bathrooms: int=None, propertytype: str=None, latitude: int=None, offset: int=None, city: str=None, state: str=None, limit: int=None, zipcode: str=None, bedrooms: int=None, longitude: int=None, address: str='5500 Grand Lake Dr, San Antonio, TX, 78244', radius: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for property records in a geographical area, or around a specific address. Returns the property features, owner details, tax assessment, and other property information. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    bathrooms: The number of bathrooms, used to search for properties matching this criteria. Supports fractions to indicate partial bathrooms
        propertytype: The type of the property, used to search for properties matching this criteria. Supported values are: `Single Family`, `Condo`, `Townhouse`, `Manufactured`, `Duplex-Triplex`, `Apartment`, `Land`
        latitude: The latitude of the search area. Use the `latitude`/`longitude` and `radius` parameters to search for properties in a specific area
        offset: The index of the first property record to return, used to paginate through large lists of results. Defaults to 0 if not provided
        city: The name of the city, used to search for properties in a specific city
        state: The 2-character state abbreviation, used to search for properties in a specific state
        limit: The maximum number of property records to return, between 1 and 500. Defaults to 50 if not provided
        zipcode: The 5-digit zip code, used to search for properties in a specific zip code
        bedrooms: The number of bedrooms, used to search for properties matching this criteria
        longitude: The longitude of the search area. Use the `latitude`/`longitude` and `radius` parameters to search for properties in a specific area
        address: The address of the property, in the format of 'Street, City, State, Zip'. Used to retrieve data for a specific property, or together with the `radius` parameter to search for properties in a specific area
        radius: The radius of the search area, in kilometers. Use in combination with the `latitude`/`longitude` or `address` parameters to search for properties in a specific area
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/properties"
    querystring = {}
    if bathrooms:
        querystring['bathrooms'] = bathrooms
    if propertytype:
        querystring['propertyType'] = propertytype
    if latitude:
        querystring['latitude'] = latitude
    if offset:
        querystring['offset'] = offset
    if city:
        querystring['city'] = city
    if state:
        querystring['state'] = state
    if limit:
        querystring['limit'] = limit
    if zipcode:
        querystring['zipCode'] = zipcode
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if longitude:
        querystring['longitude'] = longitude
    if address:
        querystring['address'] = address
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_rent_estimate(squarefootage: int=1600, bedrooms: int=4, longitude: int=None, propertytype: str='Single Family', address: str='5500 Grand Lake Drive, San Antonio, TX, 78244', maxradius: int=None, bathrooms: int=2, daysold: int=None, compcount: int=5, latitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a rent estimate and comparable rental listings for an address or lat/long coordinate. Providing property feature parameters will improve the estimate accuracy. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    squarefootage: The total living area size of the property, in square feet
        bedrooms: The number of bedrooms in the property
        longitude: The longitude of the property. The `latitude`/`longitude` can be provided instead of the `address` parameter
        propertytype: The type of the property. Supported values are: `Single Family`, `Condo`, `Townhouse`, `Manufactured`, `Duplex-Triplex`, `Apartment`
        address: The property address, in the format of 'Street, City, State, Zip'. You need to provide either the `address` or the `latitude`/`longitude` parameters
        maxradius: The maximum distance between comparable listings and the subject property, in kilometers. Defaults to 50 if not provided
        bathrooms: The number of bathrooms in the property. Supports fractions to indicate partial bathrooms
        daysold: The maximum number of days since comparable listings were last seen on the market, with a minimum of 1
        compcount: The number of comparable listings returned by the API, between 5 and 25. Defaults to 10 if not provided
        latitude: The latitude of the property. The `latitude`/`longitude` can be provided instead of the `address` parameter
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/rentalPrice"
    querystring = {}
    if squarefootage:
        querystring['squareFootage'] = squarefootage
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if longitude:
        querystring['longitude'] = longitude
    if propertytype:
        querystring['propertyType'] = propertytype
    if address:
        querystring['address'] = address
    if maxradius:
        querystring['maxRadius'] = maxradius
    if bathrooms:
        querystring['bathrooms'] = bathrooms
    if daysold:
        querystring['daysOld'] = daysold
    if compcount:
        querystring['compCount'] = compcount
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_property_records(limit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of property records selected at random up to the specified limit, to facilitate applications requiring samples of randomly selected property data. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    limit: The number of records to return, between 1 and 500. Defaults to 10 if not provided
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/randomProperties"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_market_data(zipcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns rental data for a single US zip code, including rent averages, rental listing composition, and historical rent trends. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    zipcode: A valid 5-digit US zip code
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/zipCodes/{zipcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_value_estimate(bedrooms: int=4, daysold: int=None, bathrooms: int=2, maxradius: int=None, latitude: int=None, squarefootage: int=1600, propertytype: str='Single Family', longitude: int=None, address: str='5500 Grand Lake Drive, San Antonio, TX, 78244', compcount: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a property value estimate and comparable sale listings for an address or lat/long coordinate. Providing property feature parameters will improve the estimate accuracy. [More info.](https://rapidapi.com/realtymole/api/realty-mole-property-api/details)"
    bedrooms: The number of bedrooms in the property
        daysold: The maximum number of days since comparable listings were last seen on the market, with a minimum of 1
        bathrooms: The number of bathrooms in the property. Supports fractions to indicate partial bathrooms
        maxradius: The maximum distance between comparable listings and the subject property, in kilometers. Defaults to 50 if not provided
        latitude: The latitude of the property. The `latitude`/`longitude` can be provided instead of the `address` parameter
        squarefootage: The total living area size of the property, in square feet
        propertytype: The type of the property. Supported values are: `Single Family`, `Condo`, `Townhouse`, `Manufactured`, `Duplex-Triplex`, `Apartment`, `Land`
        longitude: The longitude of the property. The `latitude`/`longitude` can be provided instead of the `address` parameter
        address: The property address, in the format of 'Street, City, State, Zip'. You need to provide either the `address` or the `latitude`/`longitude` parameters
        compcount: The number of comparable listings returned by the API, between 5 and 25. Defaults to 10 if not provided
        
    """
    url = f"https://realty-mole-property-api.p.rapidapi.com/salePrice"
    querystring = {}
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if daysold:
        querystring['daysOld'] = daysold
    if bathrooms:
        querystring['bathrooms'] = bathrooms
    if maxradius:
        querystring['maxRadius'] = maxradius
    if latitude:
        querystring['latitude'] = latitude
    if squarefootage:
        querystring['squareFootage'] = squarefootage
    if propertytype:
        querystring['propertyType'] = propertytype
    if longitude:
        querystring['longitude'] = longitude
    if address:
        querystring['address'] = address
    if compcount:
        querystring['compCount'] = compcount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

