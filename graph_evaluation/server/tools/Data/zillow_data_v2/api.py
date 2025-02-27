import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_coordinates(north: str, west: str, east: str, south: str, min_lot_size: int=None, has_view_mountain: bool=None, has_view_water: bool=None, has_waterfront: bool=None, keywords: str=None, has_view_park: bool=None, days_on_zillow: str=None, max_year_built: int=None, has_view_city: bool=None, has_single_story_only: bool=None, has_basement_finished: bool=None, has_55_plus_communities: bool=None, max_lot_size: int=None, has_pool: bool=None, home_type: str=None, min_square_feet: int=None, has_ac: bool=None, min_year_built: int=None, max_bathroom: int=None, max_square_feet: int=None, has_basement_unfinished: bool=None, has_incomplete_hoa: bool=None, has_3d_tour: bool=None, max_hoa: str=None, min_bedroom: int=None, has_open_house: bool=None, has_garage: bool=None, status: str=None, min_price: int=None, min_parking_spots: int=None, page: int=1, max_bedroom: int=None, max_price: int=None, min_bathroom: int=None, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by coordinates"
    west:   
        days_on_zillow: `days_on_zillow`: Days On Zillow
        home_type: Property type comma-separated or empty for all types:

- `Houses`: Houses
- `Townhomes`: Townhomes
- `MultiFamily`: Multi-family
- `CondosOrCoops`: Condos/Co-ops
- `LotsLand`: Lots/Land
- `Apartments`: Apartments
- `Manufactured`: Manufactured
        max_bathroom: - `max_bathroom`: MAX Bathrooms
        has_incomplete_hoa: **Include homes with incomplete HOA data**
This option is available when max_hoa is selected and the option is not Any/Empty/Do not include in the request.
        max_hoa: **Homeowners Association (HOA)**

- `Any`: Any
- `0`: No HOA Fee
- `50`: $50/month
- `100`: $100/month
- `200`: $200/month
- `300`: $300/month
- `400`: $400/month
- `500`: $500/month
- `600`: $600/month
- `700`: $700/month
- `800`: $800/month
- `900`: $900/month
- `1000` $1000/month
        min_bedroom: - `min_bedroom`: MIN Bedrooms
        status: - `ForSale`: For Sale (Default value)

- `ForRent`: Ror Rent

- `RecentlySold`: Sold
        min_price: `min_price`: MIN price
        page: `page`: Page must be larger than 0
        max_bedroom: - `max_bedroom`: MAX Bedrooms
        max_price: `max_price`: Max Price
        min_bathroom: - `min_bathroom`: MIN Bathrooms
        sort_by: - `HomesForYou`: Homes for You (Default value)
- `PriceHighToLow`: Price (High to Low)
- `PriceLowToHigh`: Price (Low to High)
- `Newest`: Newest
- `Bedrooms`: Bedrooms
- `Bathrooms`: Bathrooms
- `SquareFeet`: Square Feet
- `LotSize`: Lot Size
        
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/search-coordinates"
    querystring = {'north': north, 'west': west, 'east': east, 'south': south, }
    if min_lot_size:
        querystring['min_lot_size'] = min_lot_size
    if has_view_mountain:
        querystring['has_view_mountain'] = has_view_mountain
    if has_view_water:
        querystring['has_view_water'] = has_view_water
    if has_waterfront:
        querystring['has_waterfront'] = has_waterfront
    if keywords:
        querystring['keywords'] = keywords
    if has_view_park:
        querystring['has_view_park'] = has_view_park
    if days_on_zillow:
        querystring['days_on_zillow'] = days_on_zillow
    if max_year_built:
        querystring['max_year_built'] = max_year_built
    if has_view_city:
        querystring['has_view_city'] = has_view_city
    if has_single_story_only:
        querystring['has_single_story_only'] = has_single_story_only
    if has_basement_finished:
        querystring['has_basement_finished'] = has_basement_finished
    if has_55_plus_communities:
        querystring['has_55_plus_communities'] = has_55_plus_communities
    if max_lot_size:
        querystring['max_lot_size'] = max_lot_size
    if has_pool:
        querystring['has_pool'] = has_pool
    if home_type:
        querystring['home_type'] = home_type
    if min_square_feet:
        querystring['min_square_feet'] = min_square_feet
    if has_ac:
        querystring['has_ac'] = has_ac
    if min_year_built:
        querystring['min_year_built'] = min_year_built
    if max_bathroom:
        querystring['max_bathroom'] = max_bathroom
    if max_square_feet:
        querystring['max_square_feet'] = max_square_feet
    if has_basement_unfinished:
        querystring['has_basement_unfinished'] = has_basement_unfinished
    if has_incomplete_hoa:
        querystring['has_incomplete_hoa'] = has_incomplete_hoa
    if has_3d_tour:
        querystring['has_3d_tour'] = has_3d_tour
    if max_hoa:
        querystring['max_hoa'] = max_hoa
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if has_open_house:
        querystring['has_open_house'] = has_open_house
    if has_garage:
        querystring['has_garage'] = has_garage
    if status:
        querystring['status'] = status
    if min_price:
        querystring['min_price'] = min_price
    if min_parking_spots:
        querystring['min_parking_spots'] = min_parking_spots
    if page:
        querystring['page'] = page
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if max_price:
        querystring['max_price'] = max_price
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_details(zpid: str='78842146', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property details"
    zpid: `zpid`:  zpid 
        
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/property/detail-v3"
    querystring = {}
    if zpid:
        querystring['zpid'] = zpid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_url(url: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by URL"
    url: `url`: Must have searchQueryState in the URL
For example:
https://www.zillow.com/tokio-tx/houses/?searchQueryState=%7B%22pagination%22..........
        
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/property/url"
    querystring = {'url': url, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(location: str, min_square_feet: int=None, has_basement_finished: bool=None, min_year_built: int=None, has_garage: bool=None, has_3d_tour: bool=None, has_incomplete_hoa: bool=None, has_open_house: bool=None, page: int=1, sort_by: str=None, has_view_park: bool=None, has_waterfront: bool=None, has_55_plus_communities: bool=None, keywords: str=None, has_ac: bool=None, has_view_water: bool=None, has_view_city: bool=None, has_view_mountain: bool=None, has_basement_unfinished: bool=None, min_parking_spots: int=None, has_pool: bool=None, days_on_zillow: str=None, min_lot_size: int=None, has_single_story_only: bool=None, max_lot_size: int=None, max_year_built: int=None, max_square_feet: int=None, home_type: str=None, max_hoa: str=None, max_bedroom: int=None, min_price: int=None, status: str=None, min_bathroom: int=None, max_bathroom: int=None, max_price: int=None, min_bedroom: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    location: `location`: Address, neighborhood, or ZIP
        has_incomplete_hoa: **Include homes with incomplete HOA data**
This option is available when max_hoa is selected and the option is not Any/Empty/Do not include in the request.
        page: `page`: Page must be larger than 0
        sort_by: - `HomesForYou`: Homes for You (Default value)
- `PriceHighToLow`: Price (High to Low)
- `PriceLowToHigh`: Price (Low to High)
- `Newest`: Newest
- `Bedrooms`: Bedrooms
- `Bathrooms`: Bathrooms
- `SquareFeet`: Square Feet
- `LotSize`: Lot Size
        days_on_zillow: `days_on_zillow`: Days On Zillow
        home_type: Property type comma-separated or empty for all types:

- `Houses`: Houses
- `Townhomes`: Townhomes
- `MultiFamily`: Multi-family
- `CondosOrCoops`: Condos/Co-ops
- `LotsLand`: Lots/Land
- `Apartments`: Apartments
- `Manufactured`: Manufactured
        max_hoa: **Homeowners Association (HOA)**

- `Any`: Any
- `0`: No HOA Fee
- `50`: $50/month
- `100`: $100/month
- `200`: $200/month
- `300`: $300/month
- `400`: $400/month
- `500`: $500/month
- `600`: $600/month
- `700`: $700/month
- `800`: $800/month
- `900`: $900/month
- `1000` $1000/month
        max_bedroom: - `max_bedroom`: MAX Bedrooms
        min_price: `min_price`: MIN price
        status: - `ForSale`: For Sale (Default value)

- `ForRent`: Ror Rent

- `RecentlySold`: Sold
        min_bathroom: - `min_bathroom`: MIN Bathrooms
        max_bathroom: - `max_bathroom`: MAX Bathrooms
        max_price: `max_price`: Max Price
        min_bedroom: - `min_bedroom`: MIN Bedrooms
        
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/search"
    querystring = {'location': location, }
    if min_square_feet:
        querystring['min_square_feet'] = min_square_feet
    if has_basement_finished:
        querystring['has_basement_finished'] = has_basement_finished
    if min_year_built:
        querystring['min_year_built'] = min_year_built
    if has_garage:
        querystring['has_garage'] = has_garage
    if has_3d_tour:
        querystring['has_3d_tour'] = has_3d_tour
    if has_incomplete_hoa:
        querystring['has_incomplete_hoa'] = has_incomplete_hoa
    if has_open_house:
        querystring['has_open_house'] = has_open_house
    if page:
        querystring['page'] = page
    if sort_by:
        querystring['sort_by'] = sort_by
    if has_view_park:
        querystring['has_view_park'] = has_view_park
    if has_waterfront:
        querystring['has_waterfront'] = has_waterfront
    if has_55_plus_communities:
        querystring['has_55_plus_communities'] = has_55_plus_communities
    if keywords:
        querystring['keywords'] = keywords
    if has_ac:
        querystring['has_ac'] = has_ac
    if has_view_water:
        querystring['has_view_water'] = has_view_water
    if has_view_city:
        querystring['has_view_city'] = has_view_city
    if has_view_mountain:
        querystring['has_view_mountain'] = has_view_mountain
    if has_basement_unfinished:
        querystring['has_basement_unfinished'] = has_basement_unfinished
    if min_parking_spots:
        querystring['min_parking_spots'] = min_parking_spots
    if has_pool:
        querystring['has_pool'] = has_pool
    if days_on_zillow:
        querystring['days_on_zillow'] = days_on_zillow
    if min_lot_size:
        querystring['min_lot_size'] = min_lot_size
    if has_single_story_only:
        querystring['has_single_story_only'] = has_single_story_only
    if max_lot_size:
        querystring['max_lot_size'] = max_lot_size
    if max_year_built:
        querystring['max_year_built'] = max_year_built
    if max_square_feet:
        querystring['max_square_feet'] = max_square_feet
    if home_type:
        querystring['home_type'] = home_type
    if max_hoa:
        querystring['max_hoa'] = max_hoa
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if min_price:
        querystring['min_price'] = min_price
    if status:
        querystring['status'] = status
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if max_bathroom:
        querystring['max_bathroom'] = max_bathroom
    if max_price:
        querystring['max_price'] = max_price
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find location for search"
    text: `text`: Input to search location
        
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/location"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def photos(zpid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Photos"
    
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/extend/photos"
    querystring = {'zpid': zpid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar_results_nearby(location: str, has_basement_unfinished: bool=None, days_on_zillow: str=None, has_view_water: bool=None, has_basement_finished: bool=None, has_view_park: bool=None, min_year_built: int=None, has_view_city: bool=None, has_pool: bool=None, has_waterfront: bool=None, has_view_mountain: bool=None, keywords: str=None, has_ac: bool=None, has_single_story_only: bool=None, max_square_feet: int=None, has_55_plus_communities: bool=None, has_garage: bool=None, min_lot_size: int=None, max_year_built: int=None, max_lot_size: int=None, has_3d_tour: bool=None, min_parking_spots: int=None, min_square_feet: int=None, home_type: str=None, has_incomplete_hoa: bool=None, status: str=None, has_open_house: bool=None, min_price: int=None, max_bathroom: int=None, sort_by: str=None, min_bathroom: int=None, max_hoa: str=None, min_bedroom: int=None, max_price: int=None, page: int=1, max_bedroom: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Similar results nearby"
    location: `location`: Address, neighborhood, or ZIP
        days_on_zillow: `days_on_zillow`: Days On Zillow
        home_type: Property type comma-separated or empty for all types:

- `Houses`: Houses
- `Townhomes`: Townhomes
- `MultiFamily`: Multi-family
- `CondosOrCoops`: Condos/Co-ops
- `LotsLand`: Lots/Land
- `Apartments`: Apartments
- `Manufactured`: Manufactured
        has_incomplete_hoa: **Include homes with incomplete HOA data**
This option is available when max_hoa is selected and the option is not Any/Empty/Do not include in the request.
        status: - `ForSale`: For Sale (Default value)

- `ForRent`: Ror Rent

- `RecentlySold`: Sold
        min_price: `min_price`: MIN price
        max_bathroom: - `max_bathroom`: MAX Bathrooms
        sort_by: - `HomesForYou`: Homes for You (Default value)
- `PriceHighToLow`: Price (High to Low)
- `PriceLowToHigh`: Price (Low to High)
- `Newest`: Newest
- `Bedrooms`: Bedrooms
- `Bathrooms`: Bathrooms
- `SquareFeet`: Square Feet
- `LotSize`: Lot Size
        min_bathroom: - `min_bathroom`: MIN Bathrooms
        max_hoa: **Homeowners Association (HOA)**

- `Any`: Any
- `0`: No HOA Fee
- `50`: $50/month
- `100`: $100/month
- `200`: $200/month
- `300`: $300/month
- `400`: $400/month
- `500`: $500/month
- `600`: $600/month
- `700`: $700/month
- `800`: $800/month
- `900`: $900/month
- `1000` $1000/month
        min_bedroom: - `min_bedroom`: MIN Bedrooms
        max_price: `max_price`: Max Price
        page: `page`: Page must be larger than 0
        max_bedroom: - `max_bedroom`: MAX Bedrooms
        
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/extend/search-similar"
    querystring = {'location': location, }
    if has_basement_unfinished:
        querystring['has_basement_unfinished'] = has_basement_unfinished
    if days_on_zillow:
        querystring['days_on_zillow'] = days_on_zillow
    if has_view_water:
        querystring['has_view_water'] = has_view_water
    if has_basement_finished:
        querystring['has_basement_finished'] = has_basement_finished
    if has_view_park:
        querystring['has_view_park'] = has_view_park
    if min_year_built:
        querystring['min_year_built'] = min_year_built
    if has_view_city:
        querystring['has_view_city'] = has_view_city
    if has_pool:
        querystring['has_pool'] = has_pool
    if has_waterfront:
        querystring['has_waterfront'] = has_waterfront
    if has_view_mountain:
        querystring['has_view_mountain'] = has_view_mountain
    if keywords:
        querystring['keywords'] = keywords
    if has_ac:
        querystring['has_ac'] = has_ac
    if has_single_story_only:
        querystring['has_single_story_only'] = has_single_story_only
    if max_square_feet:
        querystring['max_square_feet'] = max_square_feet
    if has_55_plus_communities:
        querystring['has_55_plus_communities'] = has_55_plus_communities
    if has_garage:
        querystring['has_garage'] = has_garage
    if min_lot_size:
        querystring['min_lot_size'] = min_lot_size
    if max_year_built:
        querystring['max_year_built'] = max_year_built
    if max_lot_size:
        querystring['max_lot_size'] = max_lot_size
    if has_3d_tour:
        querystring['has_3d_tour'] = has_3d_tour
    if min_parking_spots:
        querystring['min_parking_spots'] = min_parking_spots
    if min_square_feet:
        querystring['min_square_feet'] = min_square_feet
    if home_type:
        querystring['home_type'] = home_type
    if has_incomplete_hoa:
        querystring['has_incomplete_hoa'] = has_incomplete_hoa
    if status:
        querystring['status'] = status
    if has_open_house:
        querystring['has_open_house'] = has_open_house
    if min_price:
        querystring['min_price'] = min_price
    if max_bathroom:
        querystring['max_bathroom'] = max_bathroom
    if sort_by:
        querystring['sort_by'] = sort_by
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if max_hoa:
        querystring['max_hoa'] = max_hoa
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if max_price:
        querystring['max_price'] = max_price
    if page:
        querystring['page'] = page
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_zip_code_by_usps(usps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get zip code by usps"
    
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/zipcode-by-usps"
    querystring = {'usps': usps, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_zip_code_by_city(city: str='Moody', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get zip code by city"
    
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/zipcode-by-city"
    querystring = {}
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_zip_code_by_state(state: str='Wyoming', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get zip code by state"
    
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/zipcode-by-state"
    querystring = {}
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_u_s_a_states(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of states"
    
    """
    url = f"https://zillow-data-v2.p.rapidapi.com/all-state"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

