import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def agents_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "agents/detail"
    id: id: Get from agents/list API
        
    """
    url = f"https://realtor-search.p.rapidapi.com/agents/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_list(postal_code: str, sort: str=None, min_price: int=None, agent_rating_min: int=None, max_price: int=None, name: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "agents/list"
    
    """
    url = f"https://realtor-search.p.rapidapi.com/agents/list"
    querystring = {'postal_code': postal_code, }
    if sort:
        querystring['sort'] = sort
    if min_price:
        querystring['min_price'] = min_price
    if agent_rating_min:
        querystring['agent_rating_min'] = agent_rating_min
    if max_price:
        querystring['max_price'] = max_price
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_photos(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get photos"
    property_id: property_id: Get from sale/rent API
        
    """
    url = f"https://realtor-search.p.rapidapi.com/property/get-photos"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_similar_homes(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search similar homes"
    property_id: property_id: Get from sale/rent API
        
    """
    url = f"https://realtor-search.p.rapidapi.com/property/search-similar"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_details(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property details"
    property_id: property_id: Get from sale/rent API
        
    """
    url = f"https://realtor-search.p.rapidapi.com/property/detail"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_rent(search_within_x_miles: int, location: str, page: int=None, keyword_search: str=None, max_sqft: int=None, has_parking: bool=None, has_gym: bool=None, min_sqft: int=None, has_laundry_room: bool=None, not_accepting_app: bool=None, has_central_air: bool=None, has_3d_tour: bool=None, has_cat: bool=None, max_bedroom: int=None, has_pool: bool=None, property_type: str=None, min_bathroom: int=None, move_in_date: str=None, max_bathroom: int=None, min_bedroom: int=None, sort_by: str=None, min_price: int=None, has_washer_dryer: bool=None, has_dog: bool=None, max_price: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for rent"
    location: Address, School, City, Zip or Neighborhood
        page: Page must be larger than 0
        keyword_search: Ex: Pool, Basement ...
        move_in_date: Format: YYYY-MM-DD
        
    """
    url = f"https://realtor-search.p.rapidapi.com/property/search-rent"
    querystring = {'search_within_x_miles': search_within_x_miles, 'location': location, }
    if page:
        querystring['page'] = page
    if keyword_search:
        querystring['keyword_search'] = keyword_search
    if max_sqft:
        querystring['max_sqft'] = max_sqft
    if has_parking:
        querystring['has_parking'] = has_parking
    if has_gym:
        querystring['has_gym'] = has_gym
    if min_sqft:
        querystring['min_sqft'] = min_sqft
    if has_laundry_room:
        querystring['has_laundry_room'] = has_laundry_room
    if not_accepting_app:
        querystring['not_accepting_app'] = not_accepting_app
    if has_central_air:
        querystring['has_central_air'] = has_central_air
    if has_3d_tour:
        querystring['has_3d_tour'] = has_3d_tour
    if has_cat:
        querystring['has_cat'] = has_cat
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if has_pool:
        querystring['has_pool'] = has_pool
    if property_type:
        querystring['property_type'] = property_type
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if move_in_date:
        querystring['move_in_date'] = move_in_date
    if max_bathroom:
        querystring['max_bathroom'] = max_bathroom
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if sort_by:
        querystring['sort_by'] = sort_by
    if min_price:
        querystring['min_price'] = min_price
    if has_washer_dryer:
        querystring['has_washer_dryer'] = has_washer_dryer
    if has_dog:
        querystring['has_dog'] = has_dog
    if max_price:
        querystring['max_price'] = max_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_sale(search_within_x_miles: int, location: str, has_golf_course: bool=None, has_clubhouses: bool=None, has_spa_hot_tubs: bool=None, has_lake_view: bool=None, has_golf_course_lot: bool=None, has_corner_lot: bool=None, has_river_view: bool=None, has_cul_de_sac: bool=None, has_pool: bool=None, has_waterfront: bool=None, has_ocean_view: bool=None, has_hill_mtn_view: bool=None, garage: str=None, max_lot_size: int=None, min_home_size: int=None, min_year_built: int=None, max_year_built: int=None, keyword_search: str=None, max_home_size: int=None, days_on_realtor: str=None, has_price_reduced: bool=None, has_incomplete_hoa_data: bool=None, listing_status: str=None, has_open_house: bool=None, sort_by: str=None, has_virtual_tours: bool=None, has_hide_foreclosures: bool=None, max_hoa: int=None, has_fireplace: bool=None, has_central_air: bool=None, stories: str=None, has_tennis_courts: bool=None, has_security_features: bool=None, has_boat_facilities: bool=None, has_spa_hot_tub: bool=None, has_horse_facilities: bool=None, has_pools: bool=None, has_disability_features: bool=None, has_rv_boat_parking: bool=None, has_dining_room: bool=None, has_hardwood_floors: bool=None, has_forced_air: bool=None, has_den_office: bool=None, has_family_room: bool=None, has_basement: bool=None, has_central_heat: bool=None, has_carport: bool=None, min_lot_size: int=None, has_3d_tours: bool=None, min_bedroom: int=None, max_bathroom: int=None, max_bedroom: int=None, min_bathroom: int=None, has_hide_pending_contingent: bool=None, property_type: str=None, max_price: int=None, min_price: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for sale"
    location: Address, School, City, Zip or Neighborhood
        min_year_built: Format: YYYY
        max_year_built: Format: YYYY
        keyword_search: Ex: Pool, Basement, Pond, Gated ...
        page: Page must be larger than 0
        
    """
    url = f"https://realtor-search.p.rapidapi.com/property/search-sale"
    querystring = {'search_within_x_miles': search_within_x_miles, 'location': location, }
    if has_golf_course:
        querystring['has_golf_course'] = has_golf_course
    if has_clubhouses:
        querystring['has_clubhouses'] = has_clubhouses
    if has_spa_hot_tubs:
        querystring['has_spa_hot_tubs'] = has_spa_hot_tubs
    if has_lake_view:
        querystring['has_lake_view'] = has_lake_view
    if has_golf_course_lot:
        querystring['has_golf_course_lot'] = has_golf_course_lot
    if has_corner_lot:
        querystring['has_corner_lot'] = has_corner_lot
    if has_river_view:
        querystring['has_river_view'] = has_river_view
    if has_cul_de_sac:
        querystring['has_cul_de_sac'] = has_cul_de_sac
    if has_pool:
        querystring['has_pool'] = has_pool
    if has_waterfront:
        querystring['has_waterfront'] = has_waterfront
    if has_ocean_view:
        querystring['has_ocean_view'] = has_ocean_view
    if has_hill_mtn_view:
        querystring['has_hill_mtn_view'] = has_hill_mtn_view
    if garage:
        querystring['garage'] = garage
    if max_lot_size:
        querystring['max_lot_size'] = max_lot_size
    if min_home_size:
        querystring['min_home_size'] = min_home_size
    if min_year_built:
        querystring['min_year_built'] = min_year_built
    if max_year_built:
        querystring['max_year_built'] = max_year_built
    if keyword_search:
        querystring['keyword_search'] = keyword_search
    if max_home_size:
        querystring['max_home_size'] = max_home_size
    if days_on_realtor:
        querystring['days_on_realtor'] = days_on_realtor
    if has_price_reduced:
        querystring['has_price_reduced'] = has_price_reduced
    if has_incomplete_hoa_data:
        querystring['has_incomplete_hoa_data'] = has_incomplete_hoa_data
    if listing_status:
        querystring['listing_status'] = listing_status
    if has_open_house:
        querystring['has_open_house'] = has_open_house
    if sort_by:
        querystring['sort_by'] = sort_by
    if has_virtual_tours:
        querystring['has_virtual_tours'] = has_virtual_tours
    if has_hide_foreclosures:
        querystring['has_hide_foreclosures'] = has_hide_foreclosures
    if max_hoa:
        querystring['max_hoa'] = max_hoa
    if has_fireplace:
        querystring['has_fireplace'] = has_fireplace
    if has_central_air:
        querystring['has_central_air'] = has_central_air
    if stories:
        querystring['stories'] = stories
    if has_tennis_courts:
        querystring['has_tennis_courts'] = has_tennis_courts
    if has_security_features:
        querystring['has_security_features'] = has_security_features
    if has_boat_facilities:
        querystring['has_boat_facilities'] = has_boat_facilities
    if has_spa_hot_tub:
        querystring['has_spa_hot_tub'] = has_spa_hot_tub
    if has_horse_facilities:
        querystring['has_horse_facilities'] = has_horse_facilities
    if has_pools:
        querystring['has_pools'] = has_pools
    if has_disability_features:
        querystring['has_disability_features'] = has_disability_features
    if has_rv_boat_parking:
        querystring['has_rv_boat_parking'] = has_rv_boat_parking
    if has_dining_room:
        querystring['has_dining_room'] = has_dining_room
    if has_hardwood_floors:
        querystring['has_hardwood_floors'] = has_hardwood_floors
    if has_forced_air:
        querystring['has_forced_air'] = has_forced_air
    if has_den_office:
        querystring['has_den_office'] = has_den_office
    if has_family_room:
        querystring['has_family_room'] = has_family_room
    if has_basement:
        querystring['has_basement'] = has_basement
    if has_central_heat:
        querystring['has_central_heat'] = has_central_heat
    if has_carport:
        querystring['has_carport'] = has_carport
    if min_lot_size:
        querystring['min_lot_size'] = min_lot_size
    if has_3d_tours:
        querystring['has_3d_tours'] = has_3d_tours
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if max_bathroom:
        querystring['max_bathroom'] = max_bathroom
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if has_hide_pending_contingent:
        querystring['has_hide_pending_contingent'] = has_hide_pending_contingent
    if property_type:
        querystring['property_type'] = property_type
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(input: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find location for search"
    input: Enter the address/Zip code you want to search for
        
    """
    url = f"https://realtor-search.p.rapidapi.com/auto-complete"
    querystring = {'input': input, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

