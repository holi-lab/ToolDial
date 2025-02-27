import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def images(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get images and photos of property."
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/getImages"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property details."
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/getDetails"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_lease(zip: str, types: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commercial Real Estate for Lease. Search by zip and type."
    types: If empty, all available types will be searched. 

Available types:
- `retail`
- `office`
- `industrial`
- `restaurant`
-  `special purpose`

For multiple selections, use comma separate values: `retail,office`
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/forLease"
    querystring = {'zip': zip, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_sale(zip: str, types: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commercial Real Estate for Sale. Search by zip and type."
    types: If empty, all available types will be searched. 

Available types:
- `retail`
- `multifamily`
- `office`
- `industrial`
-  `hospitality`
-  `mixed use`
-  `land`
-  `self storage`
-  `mobile home park`
-  `senior living`
-  `special purpose`
-  `note/loan`

For multiple selections, use comma separate values: `retail,multifamily`

        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/forSale"
    querystring = {'zip': zip, }
    if types:
        querystring['types'] = types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_property_details_by_property_id_deprecated(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "DEPRECATED
		
		
		Get the property details by property_id."
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/property"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_property_history_by_property_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the property history by property_id."
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/propertyHistory"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discover_just_sold_homes_and_properties(location: str, offset: int=0, property_type: str=None, year_built_min: int=None, sort: str=None, limit: int=50, beds_min: int=None, home_size_min: int=None, baths_min: int=None, beds_max: int=None, home_size_max: int=None, lot_size_min: int=None, lot_size_max: int=None, year_built_max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover just sold homes and properties"
    location: Location: city or zip or address
        offset: Offset results, default 0. Maximum 9800
        property_type: If empty, all available types will be searched. 

- `multi_family`
- `single_family`
- `mobile`
- `land`
- `farm`
- `condos`
- `townhome`

For multiple selections, use comma separate values: `condos,multi_family`
        sort: Default `price_high_to_low`
        limit: The number of results. Maximum 200, default 50
        home_size_min: Minimum home size (sqft)
        home_size_max: Maximum home size (sqft)
        lot_size_min: Minimum lot size (sqft)
        lot_size_max: Maximum lot size (sqft)
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/sold-homes"
    querystring = {'location': location, }
    if offset:
        querystring['offset'] = offset
    if property_type:
        querystring['property_type'] = property_type
    if year_built_min:
        querystring['year_built_min'] = year_built_min
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if beds_min:
        querystring['beds_min'] = beds_min
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if baths_min:
        querystring['baths_min'] = baths_min
    if beds_max:
        querystring['beds_max'] = beds_max
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    if year_built_max:
        querystring['year_built_max'] = year_built_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_location_suggestion(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Location suggestion. Data used in other endpoints."
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/location-suggest"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_similar_homes_by_property_id(status: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar homes"
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/similarHomes"
    querystring = {'status': status, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_flood_and_noise_data_by_property_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get flood and noise data by property id"
    id: Property id
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/floodNoise"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_property_photos(is_id: int=9366731748, property_url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the property photos by `property_id` or `property_url`"
    property_url: `https://www.realtor.com/realestateandhomes-detail/11768-SW-245th-Ter_Homestead_FL_33032_M92527-64125`
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/propertyPhotos"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if property_url:
        querystring['property_url'] = property_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def performance_metrics(session_id: str, property_type: str, zip: int, eventdate: str, listingid: int, listingprice: int, range: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property Performance Metrics"
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/propertyPerformanceMetrics"
    querystring = {'session-id': session_id, 'property_type': property_type, 'zip': zip, 'eventDate': eventdate, 'listingId': listingid, 'listingPrice': listingprice, 'range': range, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remax_agent_listings(profile_url: str='https://www.remax.com/real-estate-agents/elizabeth-morris-satellite-beach-fl/102159435', is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Remax Agent listings"
    profile_url: **Required id or profile_url.**

URL in format  `https://www.remax.com/real-estate-agents/elizabeth-morris-satellite-beach-fl/102159435`
        is_id: **Required id or profile_url.**
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/rmax-agent-listings"
    querystring = {}
    if profile_url:
        querystring['profile_url'] = profile_url
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_listings(advertiser_id: int=1859437, type: str=None, profile_url: str=None, abbreviation: str=None, member_id: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's listings."
    advertiser_id: Required `advertiser_id` or `profile_url`.
        type: Default `all`.
        profile_url: Required `advertiser_id` or `profile_url`.

URL in format `https://www.realtor.com/realestateagents/Warren-Ted-Brown_PENSACOLA_FL_1746245_884294627`
        abbreviation: Agent's mls data from `/agent/profile`
e.g. `BONY`
        member_id: Agent's mls data from `/agent/profile`
e.g. `CO3684`
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/agent/listings"
    querystring = {}
    if advertiser_id:
        querystring['advertiser_id'] = advertiser_id
    if type:
        querystring['type'] = type
    if profile_url:
        querystring['profile_url'] = profile_url
    if abbreviation:
        querystring['abbreviation'] = abbreviation
    if member_id:
        querystring['member_id'] = member_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_property_details_v2_new(is_id: int=None, property_url: str='https://www.realtor.com/realestateandhomes-detail/11768-SW-245th-Ter_Homestead_FL_33032_M92527-64125', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the property details by `property_id` or `property_url`"
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/v2/property"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if property_url:
        querystring['property_url'] = property_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_property_details_using_mls_id_deprecated(mlsid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "DEPRECATED
		
		Get the property details using MLS id."
    mlsid: e.g., F10361904 Or 2314318
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/property-by-mls"
    querystring = {'mlsId': mlsid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_property_list_for_rent_by_parameters(location: str, limit: int=50, baths_min: int=None, baths_max: int=None, home_size_max: int=None, price_min: int=None, home_size_min: int=None, price_max: int=None, days_on: str=None, beds_max: int=None, offset: int=0, beds_min: int=None, property_type: str=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a property list for **rent** by parameters.
		Let me know if you need any other parameters."
    location: Location: city or zip or address
        limit: The number of results. Maximum 200, default 50
        offset: Offset results, default 0. Maximum 9800
        property_type: If empty, all available types will be searched. 

Available types:
- `apartment`
- `single_family`
- `condos`
- `townhome`
-  `other`

For multiple selections, use comma separate values: `condos,multi_family`
        sort: - `relevance` - **default**
- `price_high_to_low`
- `price_low_to_high`
- `number_of_photos`
- `newest`
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/for-rent"
    querystring = {'location': location, }
    if limit:
        querystring['limit'] = limit
    if baths_min:
        querystring['baths_min'] = baths_min
    if baths_max:
        querystring['baths_max'] = baths_max
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if price_min:
        querystring['price_min'] = price_min
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if price_max:
        querystring['price_max'] = price_max
    if days_on:
        querystring['days_on'] = days_on
    if beds_max:
        querystring['beds_max'] = beds_max
    if offset:
        querystring['offset'] = offset
    if beds_min:
        querystring['beds_min'] = beds_min
    if property_type:
        querystring['property_type'] = property_type
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_property_details_using_mls_id_v2_new(mlsid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the property details using MLS id."
    mlsid: e.g., F10361904 Or 2314318
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/v2/property-by-mls"
    querystring = {'mlsId': mlsid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_property_list_for_sale_by_parameters_zip_address_radius(location: str, hoa_max: int=None, lot_size_max: int=None, no_hoa_fee: bool=None, days_on: str=None, offset: int=0, beds_min: int=None, baths_max: int=None, home_size_max: int=None, baths_min: int=None, property_type: str=None, sort: str=None, beds_max: int=None, year_built_min: int=None, limit: int=50, has_3d_tours: bool=None, hide_pending_contingent: bool=None, year_built_max: int=None, new_construction: bool=None, lot_size_min: int=None, home_size_min: int=None, has_virtual_tours: bool=None, open_house: bool=None, price_max: int=None, hide_foreclosure: bool=None, outside_features: str=None, expand_search_radius: str=None, price_min: int=None, community_ammenities: str=None, price_reduced: bool=None, existing: bool=None, foreclosure: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a property list for sale by parameters.
		Let me know if you need any other parameters."
    location: Location: city or zip or address
        hoa_max: Maximum HOA fee in USD
        no_hoa_fee: `true` for properties without HOA fee only.
        offset: Offset results, default 0. Maximum 9800
        beds_min: Minimum bedrooms
        baths_max: Bathrooms
        home_size_max: Maximum home size (sqft)
        baths_min: Bathrooms
        property_type: If empty, all available types will be searched. 

Available types:
- `single_family`
- `condos`
- `townhome`
- `multi_family`
- `mobile`
- `farm`
- `land`

For multiple selections, use comma separate values: `condos,multi_family`
        sort: - `relevance` - **default**
- `price_low_to_high`
- `price_high_to_low`
- `number_of_photos`
- `newest`
- `largest_sqft`
- `price_reduced_date`
        beds_max: Maximum bedrooms
        year_built_min: Minimum home age
        limit: The number of results. Maximum 200, default 50
        year_built_max: Maximum home age
        new_construction: `true` for New construction only. Leave it blank for any
        lot_size_min: Minimum lot size in sqft.
        home_size_min: Minimum home size (sqft)
        price_max: Maximum listing price (USD)
        outside_features: - `swimming_pool`
- `spa_or_hot_tub`
- `horse_facilities`

For multiple selections, use comma separate values: `swimming_pool,spa_or_hot_tub`
        expand_search_radius: Expand search radius
        price_min: Minimum listing price (USD)
        community_ammenities: - `community_swimming_pool`
- `community_spa_or_hot_tub`
- `community_golf`
- `community_security_features`
- `community_boat_facilities`
- `tennis_court`
- `community_clubhouse`
- `senior_community`

For multiple selections, use comma separate values: `senior_community,community_clubhouse`
        existing: Existing Homes
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/for-sale"
    querystring = {'location': location, }
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    if no_hoa_fee:
        querystring['no_hoa_fee'] = no_hoa_fee
    if days_on:
        querystring['days_on'] = days_on
    if offset:
        querystring['offset'] = offset
    if beds_min:
        querystring['beds_min'] = beds_min
    if baths_max:
        querystring['baths_max'] = baths_max
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if baths_min:
        querystring['baths_min'] = baths_min
    if property_type:
        querystring['property_type'] = property_type
    if sort:
        querystring['sort'] = sort
    if beds_max:
        querystring['beds_max'] = beds_max
    if year_built_min:
        querystring['year_built_min'] = year_built_min
    if limit:
        querystring['limit'] = limit
    if has_3d_tours:
        querystring['has_3d_tours'] = has_3d_tours
    if hide_pending_contingent:
        querystring['hide_pending_contingent'] = hide_pending_contingent
    if year_built_max:
        querystring['year_built_max'] = year_built_max
    if new_construction:
        querystring['new_construction'] = new_construction
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if has_virtual_tours:
        querystring['has_virtual_tours'] = has_virtual_tours
    if open_house:
        querystring['open_house'] = open_house
    if price_max:
        querystring['price_max'] = price_max
    if hide_foreclosure:
        querystring['hide_foreclosure'] = hide_foreclosure
    if outside_features:
        querystring['outside_features'] = outside_features
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if price_min:
        querystring['price_min'] = price_min
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if price_reduced:
        querystring['price_reduced'] = price_reduced
    if existing:
        querystring['existing'] = existing
    if foreclosure:
        querystring['foreclosure'] = foreclosure
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_find(zip: str, offset: int=None, limit: int=20, types: str=None, sort: str=None, agent_type: str=None, name: str=None, agent_rating_min: int=None, recommendations_count_min: int=None, price_min: int=None, price_max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for agents, teams, and offices by zip code and name."
    offset: The offset of items to be ignored in response for paging
        limit: For paging purpose (max 20)
        agent_rating_min: Rating (max 5)
        recommendations_count_min: Number of recommendations (max 10)
        price_min: Option filter by setting min price
        price_max: Option filter by setting max price
        
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/agent/find"
    querystring = {'zip': zip, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if types:
        querystring['types'] = types
    if sort:
        querystring['sort'] = sort
    if agent_type:
        querystring['agent_type'] = agent_type
    if name:
        querystring['name'] = name
    if agent_rating_min:
        querystring['agent_rating_min'] = agent_rating_min
    if recommendations_count_min:
        querystring['recommendations_count_min'] = recommendations_count_min
    if price_min:
        querystring['price_min'] = price_min
    if price_max:
        querystring['price_max'] = price_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_recommendations(advertiser_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's recommendations"
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/agent/recommendations"
    querystring = {'advertiser_id': advertiser_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_reviews(advertiser_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent reviews"
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/agent/reviews"
    querystring = {'advertiser_id': advertiser_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_profile(nrds_id: str, advertiser_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's profile."
    
    """
    url = f"https://us-real-estate-listings.p.rapidapi.com/agent/profile"
    querystring = {'nrds_id': nrds_id, 'advertiser_id': advertiser_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

