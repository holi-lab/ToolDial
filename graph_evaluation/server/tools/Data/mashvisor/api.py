import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def baths(state: str, city: str='Houston', neighborhood_id: str=None, resource: str='airbnb', beds: int=2, address: str=None, lng: int=None, baths: int=3, home_type: str=None, lat: int=None, zip_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the location (city, zip code, neighborhood, or a street address) bathrooms revenue and occupancy breakdown."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/baths"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if resource:
        querystring['resource'] = resource
    if beds:
        querystring['beds'] = beds
    if address:
        querystring['address'] = address
    if lng:
        querystring['lng'] = lng
    if baths:
        querystring['baths'] = baths
    if home_type:
        querystring['home_type'] = home_type
    if lat:
        querystring['lat'] = lat
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_activity_data(state: str, city: str='Sunnyvale', resource: str='airbnb', lat: int=None, zip_code: str='94087', home_type: str=None, lng: int=None, neighborhood_id: str=None, address: str=None, baths: int=1, beds: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the Airbnb location rental activity performance and group for booked and unbooked nights."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/rental-activity-data"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if resource:
        querystring['resource'] = resource
    if lat:
        querystring['lat'] = lat
    if zip_code:
        querystring['zip_code'] = zip_code
    if home_type:
        querystring['home_type'] = home_type
    if lng:
        querystring['lng'] = lng
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if address:
        querystring['address'] = address
    if baths:
        querystring['baths'] = baths
    if beds:
        querystring['beds'] = beds
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_agent(is_id: str, state: str, city: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves a random agent for given non-agent user."
    id: The agent Id from the Mashvisor database.
        state: The state of the property should be provided to the api or api will throw error 404
        city: The agent city from the Mashvisor database.
        name: The agent full name from the Mashvisor database.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/agents/profile/detail"
    querystring = {'id': is_id, 'state': state, }
    if city:
        querystring['city'] = city
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_types(state: str, neighborhood_id: str=None, home_type: str=None, lng: int=-122, beds: int=2, address: str='1307 Selo Dr', zip_code: str='94087', lat: int=37, baths: int=2, city: str='Sunnyvale', resource: str='airbnb', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the Search for property types in a  location (city, zip code, neighborhood, or street address) and gets their stats."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/property-types"
    querystring = {'state': state, }
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if home_type:
        querystring['home_type'] = home_type
    if lng:
        querystring['lng'] = lng
    if beds:
        querystring['beds'] = beds
    if address:
        querystring['address'] = address
    if zip_code:
        querystring['zip_code'] = zip_code
    if lat:
        querystring['lat'] = lat
    if baths:
        querystring['baths'] = baths
    if city:
        querystring['city'] = city
    if resource:
        querystring['resource'] = resource
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_super_hosts(state: str, city: str='Los Angeles', page: int=1, zip_code: int=91342, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of all Airbnb market super hosts for a zip code or a city."
    state: The state should be provided to the api or api will throw error 404
        city: A specific city you're looking for.
        page: Page number.
        zip_code: Any postal zip code.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/super-hosts"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if page:
        querystring['page'] = page
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_top_reviewed_homes(state: str, city: str='Los Angeles', reviews_count: int=30, zip_code: int=91342, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all Airbnb top reviewed homes and most counts of reviews for a specific location: city, or a zip code"
    state: The state should be provided to the api or api will throw error 404
        city: A specific city you're looking for.
        reviews_count: Any valid integer to fetch listings counts more than the number
        zip_code: Any postal zip code.
        page: Page number
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/top-reviewed"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if reviews_count:
        querystring['reviews_count'] = reviews_count
    if zip_code:
        querystring['zip_code'] = zip_code
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_newly_listed_homes(state: str, city: str='San Francisco', page: int=1, zip_code: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all Airbnb homes that are recently listed for a specific location: city, or a zip code."
    state: The state should be provided to the api or api will throw error 404.
        city: A specific city you're looking for.
        page: Page number
        zip_code: Any postal zip code.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/newly-listed"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if page:
        querystring['page'] = page
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_historical_performance(state: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the Airbnb listing 12 historical records - nightly price, revenue, occupancy days, unbooked nights, and occupancy rate - for a specific property."
    state: The state of the property should be provided to the api or api will throw error 404.
        id: The Airbnb property record Id from the Mashvisor database
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/{is_id}/historical"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_listing_info(state: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves an Airbnb listing detailed information, reviews, photos, host, estimated rental income, rental rate, night rate , calculated occupancy rate."
    state: The state of the listing should be provided to the api or api will throw error 404
        id: The Airbnb listing Id
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/{is_id}"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_occupancy_rates(state: str, city: str='Los Angeles', zip_code: int=None, neighborhood: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For each Airbnb listing, we calculate its occupancy rate, month per month, and an annual rate, and we offer our clients a 12-month historical performance for the occupancy rates. Market occupancy rates for a zip code or a neighborhood."
    state: The state should be provided to the api or api will throw error 404.
        city: A specific city you're looking for.
        zip_code: Any postal zip code.
        neighborhood: Neighborhood id you're targeting
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/occupancy-rates"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if zip_code:
        querystring['zip_code'] = zip_code
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_property_types(state: str, zip_code: int=None, city: str='Los Angeles', neighborhood: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check all Airbnb market property types for a zip code or a neighborhood and return their counts."
    state: The state should be provided to the api or api will throw error 404.
        zip_code: Any postal zip code.
        city: A specific city you're looking for.
        neighborhood: Neighborhood id you're targeting
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/property-types"
    querystring = {'state': state, }
    if zip_code:
        querystring['zip_code'] = zip_code
    if city:
        querystring['city'] = city
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_rental_rates(state: str, source: str, city: str='Los Angeles', neighborhood: int=None, zip_code: str='90291', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint retrieves rental income rates for Airbnb or traditional way for a city, zip code, or a neighborhood, you'll be able to fetch Airbnb rental rates - short term rentals, or long term rentals, calculated based on the location Airbnb occupancy rates"
    state: State short name
        source: Targeting service to fetch estiamtes for. Possible inputs: * airbnb * traditional
        city: City name
        neighborhood: Neighborhood id you're targeting
        zip_code: Zip code value
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rental-rates"
    querystring = {'state': state, 'source': source, }
    if city:
        querystring['city'] = city
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_listings(state: str, city: str='San Francisco', zip_code: int=91342, page: int=1, neighborhood: int=None, items: int=4, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all active short term rentals - Airbnb listings -  for a specific location: city, zip code, or a neighborhood"
    state: The state should be provided to the api or api will throw error 404.
        city: A specific city you're looking for.
        zip_code: Any postal zip code.
        page: Page number
        neighborhood: Neighborhood id you're targeting
        items: Item number per page
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/active-listings"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if zip_code:
        querystring['zip_code'] = zip_code
    if page:
        querystring['page'] = page
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    if items:
        querystring['items'] = items
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_market_summary(state: str, zip_code: int=None, city: str='Los Angeles', neighborhood: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a summary an overview for a specific Airbnb market location: city, zip code, or a neighborhood"
    state: The state should be provided to the api or api will throw error 404.
        zip_code: Any postal zip code.
        city: A specific city you're looking for.
        neighborhood: Neighborhood id you're targeting
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/airbnb-property/market-summary"
    querystring = {'state': state, }
    if zip_code:
        querystring['zip_code'] = zip_code
    if city:
        querystring['city'] = city
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_neighborhood_historical_performance(is_id: int, state: str, beds: int=None, average_by: str='occupancy', percentile_rate: int=1, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an Airbnb submarket (neighborhood) short term historical performance for its listings as an array"
    id: Neighborhood id to fetch data for
        state: The state should be provided to the api or api will throw error 404.
        beds: 0 to 4 bedrooms value
        average_by: Neighborhood id you're targeting. Possible Inputs: * occupancy * price * revenue
        percentile_rate: Percentile rate
        category: AirBnB category type. Possible Inputs: * flat * house * apartment * loft
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/neighborhood/{is_id}/historical/airbnb"
    querystring = {'state': state, }
    if beds:
        querystring['beds'] = beds
    if average_by:
        querystring['average_by'] = average_by
    if percentile_rate:
        querystring['percentile_rate'] = percentile_rate
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_traditional_rental_rates(state: str, source: str, city: str='Los Angeles', zip_code: str=None, neighborhood: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint retrieves rental income rates for Airbnb or traditional way for a city, zip code, or a neighborhood, you'll be able to fetch Airbnb rental rates - short term rentals, or long term rentals, calculated based on the location Airbnb occupancy rates"
    state: State short name
        source: Targeting service to fetch estiamtes for. Possible inputs: * airbnb * traditional
        city: City name
        zip_code: Zip code value
        neighborhood: Neighborhood id you're targeting
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rental-rates"
    querystring = {'state': state, 'source': source, }
    if city:
        querystring['city'] = city
    if zip_code:
        querystring['zip_code'] = zip_code
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_traditional_neighborhood_historical_performance(state: str, is_id: int, beds: str=None, year: str='2019', month: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a submarket (neighborhood) short term historical performance for its listings as an array"
    state: The state should be provided to the api or api will throw error 404.
        id: Neighborhood id to fetch data for
        beds: 0 to 4 bedrooms value
        year: A month to fetch after
        month: A month to fetch after
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/neighborhood/{is_id}/historical/traditional"
    querystring = {'state': state, }
    if beds:
        querystring['beds'] = beds
    if year:
        querystring['year'] = year
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_traditional_property(state: str, is_id: int=5637233, mls_id: str=None, address: str=None, city: str=None, parcel_number: str=None, zip_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the traditional - long term rental - property's detailed data set stored in Mashvisor database."
    state: The state of the property should be provided to the api or api will throw error 404
        is_id: The traditional property Id from the Mashvisor database.
        mls_id: Property MLS id
        address: Property street address
        city: Property city
        parcel_number: Property parcel or APN
        zip_code: Property postal code
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/traditional-property"
    querystring = {'state': state, }
    if is_id:
        querystring['id'] = is_id
    if mls_id:
        querystring['mls_id'] = mls_id
    if address:
        querystring['address'] = address
    if city:
        querystring['city'] = city
    if parcel_number:
        querystring['parcel_number'] = parcel_number
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_airbnb_cities(state: str, page: str='1', items: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Airbnb Cities, this endpoint retrieves the cities has the highest occupancy rates with their total Airbnb active listings in a specific state."
    state: State* name, ex: NV.
        page: Page number
        items: The items to return the content for. Valid values: 10, ... etc
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/trends/cities"
    querystring = {'state': state, }
    if page:
        querystring['page'] = page
    if items:
        querystring['items'] = items
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_city_summary(state: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves a summary of airbnb properties, traditional properties, investment properties, and active neighborhoods available on Mashvisor.com for a specific ."
    state: State name, ex: NV.
        city: City name, ex: Las Vegas.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/trends/summary/{state}/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_cities(state: str, page: int=1, items: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves cities a specific state."
    state: State name, ex: NV. If ignored, it will fetch all available cities
        page: The page to return the content for. Valid values:1, ... etc.
        items: The items to return the content for. Valid values: 10, ... etc.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/city/list"
    querystring = {'state': state, }
    if page:
        querystring['page'] = page
    if items:
        querystring['items'] = items
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_neighborhoods(state: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all city available neighborhoods."
    state: State name, ex: NV.
        city: City Name, Ex: Los Angeles
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/city/neighborhoods/{state}/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_top_markets(state: str, items: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the top housing cities with their active homes for sale count in a specific state."
    state: State name, ex: NV.
        items: The items to return the content for. Valid values: 10, ... etc.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/city/top-markets"
    querystring = {'state': state, }
    if items:
        querystring['items'] = items
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_investment_performance(state: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the city investment performance, median price, airbnb listings, MLS listings, traditional listings, cap rates, rental rates, and much more"
    state: State name, ex: NV.
        city: City Name, Ex: Los Angeles
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/city/investment/{state}/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def city_top_properties(city: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the city's top investment properties performance."
    city: City Name, Ex: Los Angeles
        state: State name, ex: NV.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/city/properties/{state}/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_neighborhoods(city: str, state: str, items: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the neighborhoods has the biggest occupancy in a specific city and state."
    city: Sity name, ex: Las Vegas.
        state: State name, ex: NV.
        items: The items to return the content for. Valid values: 10, ... etc.
        page: The page to return the content for. Valid values:1, ... etc
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/trends/neighborhoods"
    querystring = {'city': city, 'state': state, }
    if items:
        querystring['items'] = items
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def neighborhood_overview(is_id: int, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves a neighborhood investment analysis and overview."
    id: Neighborhood id
        state: 	Neighborhood's state
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/neighborhood/{is_id}/bar"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_taxing(state: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the property's detailed data set stored in Mashvisor database."
    state: The state of the property should be provided to the api or api will throw error 404
        id: The property Id from the Mashvisor database.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/property/{is_id}/taxing"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_office(state: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the real estate office details for a specific office id."
    state: The state of the property should be provided to the api or api will throw error 404.
        id: The office Id from the Mashvisor database.
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/property/office/details"
    querystring = {'state': state, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_location_heatmap(type: str, ne_lat: int, sw_lng: int, ne_lng: int, state: str, sw_lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the investment performance heatmap for a specific geo area."
    type: AirbnbCoc, or listingPrice, TraditionalCoc, OccupancyRate, AirbnbRental, TraditionalRental
        ne_lat: To search to a specific geo area, north east point latitude. e.g: 34.410846062851626
        sw_lng: To search to a specific geo area, south west point longitude. e.g: -118.72974734005544
        ne_lng: To search to a specific geo area, north east point longitude. e.g: -117.99366335568044
        state: The state to search in. e.g: CA
        sw_lat: To search to a specific geo area, south west point latitude. e.g: 33.76436731683163
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/search/heatmap"
    querystring = {'type': type, 'ne_lat': ne_lat, 'sw_lng': sw_lng, 'ne_lng': ne_lng, 'state': state, 'sw_lat': sw_lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_investment_performance(is_id: int, state: str, airbnb_rental: int=None, airbnb_home_owner_insurance: int=None, traditional_rental: int=None, traditional_home_owner_insurance: int=None, interest_rate: int=0, startup_cost: int=8000, traditional_total_expenses: int=None, traditional_property_tax: int=None, is_days: int=None, traditional_management_cost: int=None, payment_type: str='loan', traditional_occupancy: int=None, traditional_maintenance_cost: int=None, loan_type: int=1, airbnb_management_cost: int=None, airbnb_maintenance_cost: int=None, max_bid: int=None, airbnb_total_expenses: int=None, airbnb_occupancy: int=None, down_payment: int=10000, airbnb_property_tax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the property's investment performance."
    id: The property Id from the Mashvisor database.
        state: The state of the property should be provided to the api or api will throw error 404.
        airbnb_rental: Monthly Airbnb rental income, ex: 2000
        airbnb_home_owner_insurance: Sets the airbnb home owner insurance cost, e.g: 83
        traditional_rental: Monthly traditional rental income, ex: 1700
        traditional_home_owner_insurance: Sets the traditional home owner insurance cost, e.g: 83
        interest_rate: Interest rate
        startup_cost: Startup cost
        traditional_total_expenses: Sets the traditional total expenses, e.g: 1900
        traditional_property_tax: Sets the traditional property tax value, e.g: 1705
        is_days: If it's set to 0, the "traditional_occupancy" is considered as a percentage, if it's 1, it's considered as num of days per year
        traditional_management_cost: Sets the traditional management cost, e.g: 130
        payment_type: loan, cash
        traditional_occupancy: num of days per year, or a percentage Based on "is_days" param, eg: 70 as a percentage, or 150 as days
        traditional_maintenance_cost: Sets the traditional maintenance cost, e.g: 250
        loan_type: Loan type
        airbnb_management_cost: Sets the airbnb management cost, e.g: 120
        airbnb_maintenance_cost: Sets the airbnb maintenance cost, e.g: 230
        max_bid: Sets the property listing price to its value
        airbnb_total_expenses: Sets the airbnb total expenses, e.g: 1700
        airbnb_occupancy: num of days per year, or a percentage Based on "is_days" param, eg: 70 as a percentage, or 150 as days
        down_payment: Down payment
        airbnb_property_tax: Sets the airbnb property tax value, e.g: 1705
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/property/{is_id}/investment"
    querystring = {'state': state, }
    if airbnb_rental:
        querystring['airbnb_rental'] = airbnb_rental
    if airbnb_home_owner_insurance:
        querystring['airbnb_home_owner_insurance'] = airbnb_home_owner_insurance
    if traditional_rental:
        querystring['traditional_rental'] = traditional_rental
    if traditional_home_owner_insurance:
        querystring['traditional_home_owner_insurance'] = traditional_home_owner_insurance
    if interest_rate:
        querystring['interest_rate'] = interest_rate
    if startup_cost:
        querystring['startup_cost'] = startup_cost
    if traditional_total_expenses:
        querystring['traditional_total_expenses'] = traditional_total_expenses
    if traditional_property_tax:
        querystring['traditional_property_tax'] = traditional_property_tax
    if is_days:
        querystring['is_days'] = is_days
    if traditional_management_cost:
        querystring['traditional_management_cost'] = traditional_management_cost
    if payment_type:
        querystring['payment_type'] = payment_type
    if traditional_occupancy:
        querystring['traditional_occupancy'] = traditional_occupancy
    if traditional_maintenance_cost:
        querystring['traditional_maintenance_cost'] = traditional_maintenance_cost
    if loan_type:
        querystring['loan_type'] = loan_type
    if airbnb_management_cost:
        querystring['airbnb_management_cost'] = airbnb_management_cost
    if airbnb_maintenance_cost:
        querystring['airbnb_maintenance_cost'] = airbnb_maintenance_cost
    if max_bid:
        querystring['max_bid'] = max_bid
    if airbnb_total_expenses:
        querystring['airbnb_total_expenses'] = airbnb_total_expenses
    if airbnb_occupancy:
        querystring['airbnb_occupancy'] = airbnb_occupancy
    if down_payment:
        querystring['down_payment'] = down_payment
    if airbnb_property_tax:
        querystring['airbnb_property_tax'] = airbnb_property_tax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airbnb_comparable_listings(state: str, is_id: int, items: int=3, sort_by: str='occupancy', pid: int=None, bedrooms: int=None, order: str='desc', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the Airbnb neighborhood's listing data set in Mashvisor database with similarity and distance regarding the target MLS property."
    state: The state of the neighborhood should be provided to the api or api will throw error 404.
        id: The neighborhood Id from the Mashvisor database.
        items: Items number
        sort_by: Sorting type. Possible input: * name * similarity * distance * address * occupancy * night_price * rental_income * num_of_baths * num_of_rooms * reviews_count
        pid: Property to fetch comparable listings for.
        bedrooms: Bedrooms number; 0 - 4
        order: Order type: desc, or asc
        page: Page number
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/neighborhood/{is_id}/airbnb/details"
    querystring = {'state': state, }
    if items:
        querystring['items'] = items
    if sort_by:
        querystring['sort_by'] = sort_by
    if pid:
        querystring['pid'] = pid
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if order:
        querystring['order'] = order
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def investment_breakdown(recurring_cost: int, state: str, startup_cost: int, source: str, is_id: int, is_days: int=0, traditional_rental: int=None, turnover_cost: int=None, max_bid: int=None, airbnb_rental: int=None, airbnb_occupancy: int=None, traditional_occupancy: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the property's investment breakdown performance for Airbnb or Traditional."
    recurring_cost: Recurring cost of the investment strategy, ex: 1435
        state: The state of the property should be provided to the api or api will throw error 404.
        startup_cost: Startup cost for the investment, e.x: 8000
        source: Defines the monthly calculations should be calculated for \"Airbnb\" or \"Traditional\"
        id: The property Id from the Mashvisor database.
        is_days: If it's set to 0, the \"traditional_occupancy\" is considered as a percentage, if it's 1, it's considered as num of days per year
        traditional_rental: Monthly traditional rental income, ex: 1700
        turnover_cost: Turnover cost
        max_bid: Sets the property listing price to its value
        airbnb_rental: Monthly Airbnb rental income, ex: 2000
        airbnb_occupancy: num of days per year, or a percentage Based on \"is_days\" param, eg: 70 as a percentage, or 150 as days
        traditional_occupancy: num of days per year, or a percentage Based on \"is_days\" param, eg: 70 as a percentage, or 150 as days
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/property/{is_id}/investment/breakdown"
    querystring = {'recurring_cost': recurring_cost, 'state': state, 'startup_cost': startup_cost, 'source': source, }
    if is_days:
        querystring['is_days'] = is_days
    if traditional_rental:
        querystring['traditional_rental'] = traditional_rental
    if turnover_cost:
        querystring['turnover_cost'] = turnover_cost
    if max_bid:
        querystring['max_bid'] = max_bid
    if airbnb_rental:
        querystring['airbnb_rental'] = airbnb_rental
    if airbnb_occupancy:
        querystring['airbnb_occupancy'] = airbnb_occupancy
    if traditional_occupancy:
        querystring['traditional_occupancy'] = traditional_occupancy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_marker(state: str, payment: str, pid: int, type: str, startupcost: int=8000, loantype: int=None, loanterm: int=None, loaninterestonlyyears: int=None, downpayment: int=None, loanarmtype: int=None, interestrate: int=None, loanarmrate: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves snapshot data about a specific property."
    state: The state of the property should be provided to the api or API will throw error 404
        payment: CASH, or LOAN
        pid: The property Id from the Mashvisor database.
        type: Investment, Airbnb, or Traditional
        loantype: The loan type, e.g: 30
        downpayment: The downpayment for mortgage calculations, e.g: 40000
        loanarmtype: 3/1, 5/1, 7/1, 10/1
        interestrate: The interest rate for mortgage, e.g: 3.51
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/property/marker"
    querystring = {'state': state, 'payment': payment, 'pid': pid, 'type': type, }
    if startupcost:
        querystring['startupCost'] = startupcost
    if loantype:
        querystring['loanType'] = loantype
    if loanterm:
        querystring['loanTerm'] = loanterm
    if loaninterestonlyyears:
        querystring['loanInterestOnlyYears'] = loaninterestonlyyears
    if downpayment:
        querystring['downPayment'] = downpayment
    if loanarmtype:
        querystring['loanArmType'] = loanarmtype
    if interestrate:
        querystring['interestRate'] = interestrate
    if loanarmrate:
        querystring['loanArmRate'] = loanarmrate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traditional_comparable_listings(is_id: int, state: str, items: int=5, min_price: int=None, sort_by: str='distance', category: int=None, pid: int=325215, max_price: int=None, page: int=1, order: str='desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the traditional neighborhood's listing data set in Mashvisor database with similarity and distance regarding the target MLS property."
    id: The neighborhood Id from the Mashvisor database.
        state: The state of the neighborhood should be provided to the api or api will throw error 404.
        items: Items number
        min_price: min_price of rental value
        sort_by: Sorting type. Possible input: * address * similarity * distance * beds * baths * price
        category: Bedrooms number
        pid: Property to fetch comparable listings for.
        max_price: max_price of rental value
        page: Page number
        order: Order type: desc, or asc
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/neighborhood/{is_id}/traditional/listing"
    querystring = {'state': state, }
    if items:
        querystring['items'] = items
    if min_price:
        querystring['min_price'] = min_price
    if sort_by:
        querystring['sort_by'] = sort_by
    if category:
        querystring['category'] = category
    if pid:
        querystring['pid'] = pid
    if max_price:
        querystring['max_price'] = max_price
    if page:
        querystring['page'] = page
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_property(state: str, zip_code: int=None, is_id: int=2430136, address: str=None, parcel_number: str=None, city: str=None, mls_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the property's detailed data set stored in Mashvisor database."
    state: The state of the property should be provided to the api or api will throw error 404
        zip_code: Property zip code
        is_id: The property Id from the Mashvisor database.
        address: Property street address
        parcel_number: Property parcel or APN
        city: Property city
        mls_id: Property MLS id
        
    """
    url = f"https://mashvisor-api.p.rapidapi.com/property"
    querystring = {'state': state, }
    if zip_code:
        querystring['zip_code'] = zip_code
    if is_id:
        querystring['id'] = is_id
    if address:
        querystring['address'] = address
    if parcel_number:
        querystring['parcel_number'] = parcel_number
    if city:
        querystring['city'] = city
    if mls_id:
        querystring['mls_id'] = mls_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insights(state: str, city: str='Sunnyvale', beds: int=3, address: str=None, zip_code: str=None, baths: int=2, resource: str='airbnb', lng: int=None, home_type: str=None, neighborhood_id: str=None, lat: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the Lookup location (city, zip code, neighborhood, or street address) and gets its insights."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/lookup"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if beds:
        querystring['beds'] = beds
    if address:
        querystring['address'] = address
    if zip_code:
        querystring['zip_code'] = zip_code
    if baths:
        querystring['baths'] = baths
    if resource:
        querystring['resource'] = resource
    if lng:
        querystring['lng'] = lng
    if home_type:
        querystring['home_type'] = home_type
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def export_comps(state: str, home_type: str=None, lng: int=None, zip_code: str='76549', city: str=None, beds: int=2, neighborhood_id: str=None, baths: int=1, resource: str='airbnb', lat: int=None, address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the export of all location (city, zip code, neighborhood, or street address) comparables used in the analysis."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/export-comps"
    querystring = {'state': state, }
    if home_type:
        querystring['home_type'] = home_type
    if lng:
        querystring['lng'] = lng
    if zip_code:
        querystring['zip_code'] = zip_code
    if city:
        querystring['city'] = city
    if beds:
        querystring['beds'] = beds
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if baths:
        querystring['baths'] = baths
    if resource:
        querystring['resource'] = resource
    if lat:
        querystring['lat'] = lat
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby_listings(state: str, resource: str='airbnb', city: str=None, address: str=None, lat: int=None, zip_code: str='76549', home_type: str=None, beds: int=2, lng: int=None, baths: int=2, neighborhood_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the top5  location (city, zip code, neighborhood, or street address) MLS listings in the area."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/nearby-listings"
    querystring = {'state': state, }
    if resource:
        querystring['resource'] = resource
    if city:
        querystring['city'] = city
    if address:
        querystring['address'] = address
    if lat:
        querystring['lat'] = lat
    if zip_code:
        querystring['zip_code'] = zip_code
    if home_type:
        querystring['home_type'] = home_type
    if beds:
        querystring['beds'] = beds
    if lng:
        querystring['lng'] = lng
    if baths:
        querystring['baths'] = baths
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_comps(state: str, address: str=None, resource: str='traditional', home_type: str=None, baths: int=3, lng: int=None, zip_code: str=None, city: str='Austin', beds: int=3, neighborhood_id: str=None, lat: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the list of all locations (city, zip code, neighborhood, or street address) comparables used in the analysis."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/list-comps"
    querystring = {'state': state, }
    if address:
        querystring['address'] = address
    if resource:
        querystring['resource'] = resource
    if home_type:
        querystring['home_type'] = home_type
    if baths:
        querystring['baths'] = baths
    if lng:
        querystring['lng'] = lng
    if zip_code:
        querystring['zip_code'] = zip_code
    if city:
        querystring['city'] = city
    if beds:
        querystring['beds'] = beds
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def beds(state: str, resource: str='airbnb', address: str=None, city: str='sunnyvale', baths: int=None, neighborhood_id: str=None, lng: int=None, beds: int=None, home_type: str=None, lat: int=None, zip_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the location (city, zip code, neighborhood, or a street address) bedrooms revenue and occupancy breakdown"
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/beds"
    querystring = {'state': state, }
    if resource:
        querystring['resource'] = resource
    if address:
        querystring['address'] = address
    if city:
        querystring['city'] = city
    if baths:
        querystring['baths'] = baths
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if lng:
        querystring['lng'] = lng
    if beds:
        querystring['beds'] = beds
    if home_type:
        querystring['home_type'] = home_type
    if lat:
        querystring['lat'] = lat
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_performance(state: str, zip_code: str=None, baths: int=None, home_type: str=None, address: str=None, lat: int=None, neighborhood_id: str=None, city: str='Sunnyvale', lng: int=None, beds: int=None, resource: str='traditional', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the location (city, zip code, neighborhood, or street address) historical performance."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/historical-performance"
    querystring = {'state': state, }
    if zip_code:
        querystring['zip_code'] = zip_code
    if baths:
        querystring['baths'] = baths
    if home_type:
        querystring['home_type'] = home_type
    if address:
        querystring['address'] = address
    if lat:
        querystring['lat'] = lat
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if city:
        querystring['city'] = city
    if lng:
        querystring['lng'] = lng
    if beds:
        querystring['beds'] = beds
    if resource:
        querystring['resource'] = resource
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def revenue_stats(state: str, city: str='Sunnyvale', zip_code: str='94087', lng: int=None, neighborhood_id: str=None, lat: int=None, baths: int=None, resource: str='traditional', home_type: str=None, address: str=None, beds: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves the revenue and occupancy statistics in a  location (city, zip code, neighborhood, or street address)."
    
    """
    url = f"https://mashvisor-api.p.rapidapi.com/rento-calculator/revenue-stats"
    querystring = {'state': state, }
    if city:
        querystring['city'] = city
    if zip_code:
        querystring['zip_code'] = zip_code
    if lng:
        querystring['lng'] = lng
    if neighborhood_id:
        querystring['neighborhood_id'] = neighborhood_id
    if lat:
        querystring['lat'] = lat
    if baths:
        querystring['baths'] = baths
    if resource:
        querystring['resource'] = resource
    if home_type:
        querystring['home_type'] = home_type
    if address:
        querystring['address'] = address
    if beds:
        querystring['beds'] = beds
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mashvisor-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

