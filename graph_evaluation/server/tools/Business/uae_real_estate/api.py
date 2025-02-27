import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_property_evaluation_districts_rent_compare_get(property_type: str, price: int, longitude: int, property_sub_type: str, latitude: int, bedrooms: int, size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns a comparative evaluation of the new listing with the average
		historical price and size in sqft of the district, within the constraints of a combination of location, property type,
		and amenities. Useful for determining if the listing is overpriced or underpriced.
		Uses the latitude and longitude to determine the district.
		
		For example, the evaluation for a 1 bedroom apartment in a district will
		be different from that of a 4 bedroom Villa in the same district.
		
		**Parameters**:
		
		| parameter | description | limits |
		| --- | --- | --- |
		| **latitude, longitude** | The coordinate points in [WGS 84 Projection (EPSG / SRID 4326)](https://en.wikipedia.org/wiki/World_Geodetic_System) | Currently Restricted to UAE |
		| **property_type** | The type of property | One of the property types from `catalogue/property_types` |
		| **property_sub_type** | The subtype of the property | One of the property subtypes from `catalogue/property_sub_types` |
		| **bedrooms** | The number of bedrooms in the property | |
		| **price** | The price of the property in `AED` as an annualized rent value | |
		| **size** | The size (area) of the property in `sqft` | |
		
		**Response Variables**:
		
		| variable | description |
		| --- | --- |
		| **city** | The city in which the district exists |
		| **district** | The district for which the evaluation is returned |
		| **property_type** | The property type (ex. `Residential`) for which the evaluation is returned |
		| **property_sub_type** | The property subtype (ex. `Apartment`) for which the evaluation is returned |
		| **bedroom_count** | The number of bedrooms for which the evaluation is returned |
		| **listing_price** | The price of the input property in `AED` as an annualized rent value |
		| **listing_size** | The size (area) of the input property in `sqft` |
		| **average_rent** | The average historical rent (in annual `AED`) of the district for the given combination of location, property type and amenities |
		| **average_size** | The average historical size (in `sqft`) of the district for the for the given combination of location, property type and amenities |
		| **rent_comparison** | The percent difference between the listing price and the average historical rent of the district |
		| **size_comparison** | The percent difference between the listing size and the average historical size of the district |"
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/districts/rent/compare"
    querystring = {'property_type': property_type, 'price': price, 'longitude': longitude, 'property_sub_type': property_sub_type, 'latitude': latitude, 'bedrooms': bedrooms, 'size': size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rental_prices_districts_rent_history_get(bedrooms: int, latitude: int, property_sub_type: str, longitude: int, property_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns the historical rental prices for a combination of location, property type,
		and amenities within a district.
		Uses the latitude and longitude to determine the district.
		
		For example, the historical pricing for a 1 bedroom apartment in a district will
		be different from that of a 4 bedroom Villa in the same district.
		
		**Parameters**:
		
		| parameter | description | limits |
		| --- | --- | --- |
		| **latitude, longitude** | The coordinate points in [WGS 84 Projection (EPSG / SRID 4326)](https://en.wikipedia.org/wiki/World_Geodetic_System) | Currently Restricted to UAE |
		| **property_type** | The type of property | One of the property types from `catalogue/property_types` |
		| **property_sub_type** | The subtype of the property | One of the property subtypes from `catalogue/property_sub_types` |
		| **bedrooms** | The number of bedrooms in the property | |
		
		**Response Variables**:
		
		| variable | description |
		| --- | --- |
		| **city** | The city in which the district exists |
		| **district** | The district for which the history is returned |
		| **property_type** | The property type (ex. `Residential`) for which the history is returned |
		| **property_sub_type** | The property subtype (ex. `Apartment`) for which the history is returned |
		| **bedroom_count** | The number of bedrooms for which the history is returned |
		| **history** | A list of historical rental prices for the given combination of location, property type, and amenities. Rolling monthly-year window from previous month. |"
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/districts/rent/history"
    querystring = {'bedrooms': bedrooms, 'latitude': latitude, 'property_sub_type': property_sub_type, 'longitude': longitude, 'property_type': property_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_description_districts_describe_get(district: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns a description of the district. Provides a high-level overview of the district.
		
		**Parameters**:
		
		| parameter | description | limits |
		| --- | --- | --- |
		| **city** | The city in which the district exists | One of the cities from `catalogue/cities` |
		| **district** | The district for which the description is to be returned | One of the districts from `catalogue/districts` |
		
		
		**Response Variables**:
		
		| variable | description |
		| --- | --- |
		| **district** | The district for which the description is returned |
		| **city** | The city in which the district exists |
		| **description** | A small paragraph description of the district |
		| **last_updated_at** | The timestamp at which the description was last updated |"
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/districts/describe"
    querystring = {'district': district, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locate_district_districts_locate_get(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns the district in which the given location exists.
		
		**Parameters**:
		
		| parameter | description | limits |
		| --- | --- | --- |
		| **latitude, longitude** | The coordinate points in [WGS 84 Projection (EPSG / SRID 4326)](https://en.wikipedia.org/wiki/World_Geodetic_System) | Currently Restricted to UAE |
		
		**Response Variables**:
		
		| variable | description |
		| --- | --- |
		| **district** | The district in which the location exists |
		| **city** | The city in which the district exists |"
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/districts/locate"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A home endpoint that returns the API home page JSON."
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_districts_catalogue_districts_get(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns a list of all supported districts in a city.
		
		**Parameters**:
		| parameter | description | limits |
		| --- | --- | --- |
		| **city** | The city for which the districts are to be returned | One of the cities from `catalogue/cities` |"
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/catalogue/districts"
    querystring = {'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_types_catalogue_property_types_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns a list of all supported property types.
		A property type is a broad classification of a property. For example, `Residential`, `Commercial`, etc."
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/catalogue/property_types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cities_catalogue_cities_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns a list of all supported cities."
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/catalogue/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_property_sub_types_catalogue_property_sub_types_get(property_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**About**:
		
		Returns a list of all supported property sub types.
		A property sub type is a more specific classification of a property. For example, `Apartment`, `Villa`, etc.
		
		**Parameters**:
		| parameter | description | limits |
		| --- | --- | --- |
		| **property_type** | The property type for which the property sub types are to be returned | One of the property types from `catalogue/property_types` |"
    
    """
    url = f"https://uae-real-estate.p.rapidapi.com/catalogue/property_sub_types"
    querystring = {'property_type': property_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uae-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

