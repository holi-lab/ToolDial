import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keywords_search_suggest(keyword_text: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get keyword search suggestion for `keyword_seach` parameters in `/for-sale` endpoint"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/keywords-search-suggest"
    querystring = {'keyword_text': keyword_text, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_by_mls_id(mls_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search properties by MLS ID"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/property-by-mls-id"
    querystring = {'mls_id': mls_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_detail_deprecated(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detail data by `property_id`"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/property-detail"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_property_detail_deprecated(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detail data by `property_id` V2"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/property-detail"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_property_detail(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detail data by `property_id`"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/v3/property-detail"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_sale_home_estimate_value(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get home estimate and historical values"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/for-sale/home-estimate-value"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_sale_other_homes_in_building(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get other homes in same building by `property_id`"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/for-sale/other-homes-in-building"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_sale_similiar_homes(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similiar homes by `property_id`"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/for-sale/similiar-homes"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_for_sale_by_zipcode(zipcode: str, hide_foreclosure: str=None, open_house: str=None, offset: int=0, expand_search_radius: str=None, no_hoa_fee: str=None, keywords: str=None, home_size_min: str=None, price_reduced: str=None, home_size_max: str=None, features_in_nyc_only: str=None, garage: str=None, inside_rooms: str=None, heating_cooling: str=None, baths_min: str=None, days_on_realtor: str=None, property_type_nyc_only: str=None, has_3d_tours: str=None, price_max: str=None, stories: str=None, beds_min: str=None, lot_size_min: str=None, baths_max: str=None, home_age_max: str=None, outside_features: str=None, lot_views: str=None, hide_pending_contingent: str=None, beds_max: str=None, limit: int=42, hoa_max: str=None, property_type: str=None, new_construction: str=None, include_nearby_areas_slug_id: str=None, sort: str=None, price_min: str=None, lot_size_max: str=None, has_virtual_tours: str=None, community_ammenities: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for-sale properties.
		**Parameters**: `zipcode, limit, offset, sort:newest price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, property_type_nyc_only, new_construction, hide_pending_contingent, has_virtual_tours, has_3d_tours, hide_foreclosure, price_reduced, open_house, keywords, no_hoa_fee, hoa_max, days_on_realtor, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, lot_size_min, lot_size_max, home_age_max, stories, garage, heating_cooling, inside_rooms, outside_features, lot_views, community_ammenities, features_in_nyc_only`"
    zipcode: zipcode
        hide_foreclosure: `true` for hide foreclosure. Leave blank for any
        open_house: `true` for properties with open house only. Leave blank for any
        offset: Offset results, default 0. Maximum 9800.
        expand_search_radius: One of the following options: `1|5|10|25|50`. Expand search by radius in miles
        no_hoa_fee: `true` for properties without HOA fee only. Leave blank for any
        keywords: Comma separated values. Get popular keywords from `/keywords-search-suggest` response
        home_size_min: One of the following options: `750|1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500`. Minimum home size in sqft
        price_reduced: `true` for properties with price reduced only. Leave blank for any
        home_size_max: One of the following options: `1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500|10000`. Maximum home size in sqft
        features_in_nyc_only: Comma separated values. One or more from following options: `furniture|dishwasher|community_doorman|pets_allowed|laundry_room|elevator|community_outdoor_space`
        garage: One of the following options: `1+|2+|3+`
        inside_rooms: Comma separated values. One or more comma separated from following options: `basement|hardwood_floors|fireplace|disability_features|den_or_office|family_room|dining_room`
        heating_cooling: Comma separated values. One or more from following options: `central_air|central_heat|forced_air`
        baths_min: Minimum bathrooms
        days_on_realtor: One of the following options: `today|7|14|21|30`
        property_type_nyc_only: Comma separated values. One or more from following options: `condo|coop|condop`. For NYC listings only
        has_3d_tours: `true` for properties with 3D tour only. Leave blank for any
        price_max: Maximum list price in USD
        stories: One of the following options: `single|multi`
        beds_min: Minimum bedrooms
        lot_size_min: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Minimum lot size in sqft
        baths_max: Maximum bathrooms
        home_age_max: Maximum home age
        outside_features: Comma separated values. One or more from following options: `swimming_pool|spa_or_hot_tub|horse_facilities`
        lot_views: Comma separated values. One or more from following options: `waterfront|cul_de_sac|corner_lot|golf_course_lot_or_frontage|hill_or_mountain_view|ocean_view|lake_view|river_view`
        hide_pending_contingent: `true` for hide pending/contingent. Leave blank for any
        beds_max: Maximum bedrooms
        limit: Number of results. Maximum 200 for Paid Plan, default 42
        hoa_max: Maximum HOA fee in USD
        property_type: Comma separated values. One or more from following options: `multi_family|single_family|mobile|land|farm`
        new_construction: `true` for New construction only. Leave blank for any
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-sale-nearby-areas`
        sort: One of the following options: `relevant|newest|lowest_price|highest_price|open_house_date|price_reduced_date|largest_sqft|lot_size|sold_date`. Default is relevant
        price_min: Minimum list price in USD
        lot_size_max: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Maximum  lot size in sqft
        has_virtual_tours: `true` for properties with virtual tour only. Leave blank for any
        community_ammenities: Comma separated values. One or more from following options: `community_swimming_pool|community_spa_or_hot_tub|community_golf|community_security_features|community_boat_facilities|tennis_court|community_clubhouse|senior_community`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/for-sale-by-zipcode"
    querystring = {'zipcode': zipcode, }
    if hide_foreclosure:
        querystring['hide_foreclosure'] = hide_foreclosure
    if open_house:
        querystring['open_house'] = open_house
    if offset:
        querystring['offset'] = offset
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if no_hoa_fee:
        querystring['no_hoa_fee'] = no_hoa_fee
    if keywords:
        querystring['keywords'] = keywords
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if price_reduced:
        querystring['price_reduced'] = price_reduced
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if features_in_nyc_only:
        querystring['features_in_nyc_only'] = features_in_nyc_only
    if garage:
        querystring['garage'] = garage
    if inside_rooms:
        querystring['inside_rooms'] = inside_rooms
    if heating_cooling:
        querystring['heating_cooling'] = heating_cooling
    if baths_min:
        querystring['baths_min'] = baths_min
    if days_on_realtor:
        querystring['days_on_realtor'] = days_on_realtor
    if property_type_nyc_only:
        querystring['property_type_nyc_only'] = property_type_nyc_only
    if has_3d_tours:
        querystring['has_3d_tours'] = has_3d_tours
    if price_max:
        querystring['price_max'] = price_max
    if stories:
        querystring['stories'] = stories
    if beds_min:
        querystring['beds_min'] = beds_min
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if baths_max:
        querystring['baths_max'] = baths_max
    if home_age_max:
        querystring['home_age_max'] = home_age_max
    if outside_features:
        querystring['outside_features'] = outside_features
    if lot_views:
        querystring['lot_views'] = lot_views
    if hide_pending_contingent:
        querystring['hide_pending_contingent'] = hide_pending_contingent
    if beds_max:
        querystring['beds_max'] = beds_max
    if limit:
        querystring['limit'] = limit
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if property_type:
        querystring['property_type'] = property_type
    if new_construction:
        querystring['new_construction'] = new_construction
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if sort:
        querystring['sort'] = sort
    if price_min:
        querystring['price_min'] = price_min
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    if has_virtual_tours:
        querystring['has_virtual_tours'] = has_virtual_tours
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_for_sale_result_count(city: str, state_code: str, has_3d_tours: str=None, property_type_nyc_only: str=None, home_age_max: str=None, baths_max: str=None, lot_size_min: str=None, stories: str=None, heating_cooling: str=None, lot_views: str=None, new_construction: str=None, community_ammenities: str=None, lot_size_max: str=None, hide_foreclosure: str=None, keywords: str=None, price_max: str=None, has_virtual_tours: str=None, location: str=None, beds_min: str=None, beds_max: str=None, days_on_realtor: str=None, home_size_min: str=None, features_in_nyc_only: str=None, no_hoa_fee: str=None, price_min: str=None, garage: str=None, open_house: str=None, price_reduced: str=None, hide_pending_contingent: str=None, baths_min: str=None, expand_search_radius: str=None, inside_rooms: str=None, home_size_max: str=None, hoa_max: str=None, include_nearby_areas_slug_id: str=None, outside_features: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get for-sale search result count.
		**Parameters**: `city, state_code, location, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, property_type_nyc_only, new_construction, hide_pending_contingent, has_virtual_tours, has_3d_tours, hide_foreclosure, price_reduced, open_house, keywords, no_hoa_fee, hoa_max, days_on_realtor, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, lot_size_min, lot_size_max, home_age_max, stories, garage, heating_cooling, inside_rooms, outside_features, lot_views, community_ammenities, features_in_nyc_only`"
    city: City name. Get data from `/location/suggest` response
        state_code: State Code. Get from `/location/suggest` response
        has_3d_tours: `true` for properties with 3D tour only. Leave blank for any
        property_type_nyc_only: Comma separated values. One or more from following options: `condo|coop|condop`. For NYC listings only
        home_age_max: Maximum home age
        baths_max: Maximum bathrooms
        lot_size_min: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Minimum lot size in sqft
        stories: One of the following options: `single|multi`
        heating_cooling: Comma separated values. One or more from following options: `central_air|central_heat|forced_air`
        lot_views: Comma separated values. One or more from following options: `waterfront|cul_de_sac|corner_lot|golf_course_lot_or_frontage|hill_or_mountain_view|ocean_view|lake_view|river_view`
        new_construction: `true` for New construction only. Leave blank for any
        community_ammenities: Comma separated values. One or more from following options: `community_swimming_pool|community_spa_or_hot_tub|community_golf|community_security_features|community_boat_facilities|tennis_court|community_clubhouse|senior_community`
        lot_size_max: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Maximum  lot size in sqft
        hide_foreclosure: `true` for hide foreclosure. Leave blank for any
        keywords: Comma separated values. Get popular keywords from `/keywords-search-suggest` response
        price_max: Maximum list price in USD
        has_virtual_tours: `true` for properties with virtual tour only. Leave blank for any
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        beds_min: Minimum bedrooms
        beds_max: Maximum bedrooms
        days_on_realtor: One of the following options: `today|7|14|21|30`
        home_size_min: One of the following options: `750|1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500`. Minimum home size in sqft
        features_in_nyc_only: Comma separated values. One or more from following options: `furniture|dishwasher|community_doorman|pets_allowed|laundry_room|elevator|community_outdoor_space`
        no_hoa_fee: `true` for properties without HOA fee only. Leave blank for any
        price_min: Minimum list price in USD
        garage: One of the following options: `1+|2+|3+`
        open_house: `true` for properties with open house only. Leave blank for any
        price_reduced: `true` for properties with price reduced only. Leave blank for any
        hide_pending_contingent: `true` for hide pending/contingent. Leave blank for any
        baths_min: Minimum bathrooms
        expand_search_radius: One of the following options: `1|5|10|25|50`. Expand search by radius in miles
        inside_rooms: Comma separated values. One or more comma separated from following options: `basement|hardwood_floors|fireplace|disability_features|den_or_office|family_room|dining_room`
        home_size_max: One of the following options: `1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500|10000`. Maximum home size in sqft
        hoa_max: Maximum HOA fee in USD
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-sale-nearby-areas`
        outside_features: Comma separated values. One or more from following options: `swimming_pool|spa_or_hot_tub|horse_facilities`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/for-sale-result-count"
    querystring = {'city': city, 'state_code': state_code, }
    if has_3d_tours:
        querystring['has_3d_tours'] = has_3d_tours
    if property_type_nyc_only:
        querystring['property_type_nyc_only'] = property_type_nyc_only
    if home_age_max:
        querystring['home_age_max'] = home_age_max
    if baths_max:
        querystring['baths_max'] = baths_max
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if stories:
        querystring['stories'] = stories
    if heating_cooling:
        querystring['heating_cooling'] = heating_cooling
    if lot_views:
        querystring['lot_views'] = lot_views
    if new_construction:
        querystring['new_construction'] = new_construction
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    if hide_foreclosure:
        querystring['hide_foreclosure'] = hide_foreclosure
    if keywords:
        querystring['keywords'] = keywords
    if price_max:
        querystring['price_max'] = price_max
    if has_virtual_tours:
        querystring['has_virtual_tours'] = has_virtual_tours
    if location:
        querystring['location'] = location
    if beds_min:
        querystring['beds_min'] = beds_min
    if beds_max:
        querystring['beds_max'] = beds_max
    if days_on_realtor:
        querystring['days_on_realtor'] = days_on_realtor
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if features_in_nyc_only:
        querystring['features_in_nyc_only'] = features_in_nyc_only
    if no_hoa_fee:
        querystring['no_hoa_fee'] = no_hoa_fee
    if price_min:
        querystring['price_min'] = price_min
    if garage:
        querystring['garage'] = garage
    if open_house:
        querystring['open_house'] = open_house
    if price_reduced:
        querystring['price_reduced'] = price_reduced
    if hide_pending_contingent:
        querystring['hide_pending_contingent'] = hide_pending_contingent
    if baths_min:
        querystring['baths_min'] = baths_min
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if inside_rooms:
        querystring['inside_rooms'] = inside_rooms
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if outside_features:
        querystring['outside_features'] = outside_features
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_for_sale(state_code: str, offset: int, city: str, limit: int, price_min: str=None, no_hoa_fee: str=None, sort: str='newest', property_type: str=None, has_3d_tours: str=None, heating_cooling: str=None, open_house: str=None, beds_min: str=None, home_size_max: str=None, property_type_nyc_only: str=None, baths_min: str=None, include_nearby_areas_slug_id: str=None, lot_size_max: str=None, garage: str=None, lot_views: str=None, home_age_max: str=None, home_size_min: str=None, inside_rooms: str=None, days_on_realtor: str=None, community_ammenities: str=None, features_in_nyc_only: str=None, stories: str=None, price_max: str=None, hide_pending_contingent: str=None, expand_search_radius: str=None, has_virtual_tours: str=None, hide_foreclosure: str=None, location: str=None, price_reduced: str=None, baths_max: str=None, beds_max: str=None, new_construction: str=None, outside_features: str=None, hoa_max: str=None, keywords: str=None, lot_size_min: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for-sale properties.
		**Parameters**: `city, state_code, location, limit, offset, sort:newest price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, property_type_nyc_only, new_construction, hide_pending_contingent, has_virtual_tours, has_3d_tours, hide_foreclosure, price_reduced, open_house, keywords, no_hoa_fee, hoa_max, days_on_realtor, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, lot_size_min, lot_size_max, home_age_max, stories, garage, heating_cooling, inside_rooms, outside_features, lot_views, community_ammenities, features_in_nyc_only`"
    state_code: State Code. Get from `/location/suggest` response
        offset: Offset results, default 0. Maximum 9800.
        city: City name. Get data from `/location/suggest` response
        limit: Number of results. Maximum 200 for Paid Plan, default 42
        price_min: Minimum list price in USD
        no_hoa_fee: `true` for properties without HOA fee only. Leave blank for any
        sort: One of the following options: `relevant|newest|lowest_price|highest_price|open_house_date|price_reduced_date|largest_sqft|lot_size|sold_date`. Default is relevant
        property_type: Comma separated values. One or more from following options: `multi_family|single_family|mobile|land|farm`
        has_3d_tours: `true` for properties with 3D tour only. Leave blank for any
        heating_cooling: Comma separated values. One or more from following options: `central_air|central_heat|forced_air`
        open_house: `true` for properties with open house only. Leave blank for any
        beds_min: Minimum bedrooms
        home_size_max: One of the following options: `1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500|10000`. Maximum home size in sqft
        property_type_nyc_only: Comma separated values. One or more from following options: `condo|coop|condop`. For NYC listings only
        baths_min: Minimum bathrooms
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-sale-nearby-areas`
        lot_size_max: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Maximum  lot size in sqft
        garage: One of the following options: `1+|2+|3+`
        lot_views: Comma separated values. One or more from following options: `waterfront|cul_de_sac|corner_lot|golf_course_lot_or_frontage|hill_or_mountain_view|ocean_view|lake_view|river_view`
        home_age_max: Maximum home age
        home_size_min: One of the following options: `750|1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500`. Minimum home size in sqft
        inside_rooms: Comma separated values. One or more comma separated from following options: `basement|hardwood_floors|fireplace|disability_features|den_or_office|family_room|dining_room`
        days_on_realtor: One of the following options: `today|7|14|21|30`
        community_ammenities: Comma separated values. One or more from following options: `community_swimming_pool|community_spa_or_hot_tub|community_golf|community_security_features|community_boat_facilities|tennis_court|community_clubhouse|senior_community`
        features_in_nyc_only: Comma separated values. One or more from following options: `furniture|dishwasher|community_doorman|pets_allowed|laundry_room|elevator|community_outdoor_space`
        stories: One of the following options: `single|multi`
        price_max: Maximum list price in USD
        hide_pending_contingent: `true` for hide pending/contingent. Leave blank for any
        expand_search_radius: One of the following options: `1|5|10|25|50`. Expand search by radius in miles
        has_virtual_tours: `true` for properties with virtual tour only. Leave blank for any
        hide_foreclosure: `true` for hide foreclosure. Leave blank for any
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        price_reduced: `true` for properties with price reduced only. Leave blank for any
        baths_max: Maximum bathrooms
        beds_max: Maximum bedrooms
        new_construction: `true` for New construction only. Leave blank for any
        outside_features: Comma separated values. One or more from following options: `swimming_pool|spa_or_hot_tub|horse_facilities`
        hoa_max: Maximum HOA fee in USD
        keywords: Comma separated values. Get popular keywords from `/keywords-search-suggest` response
        lot_size_min: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Minimum lot size in sqft
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/for-sale"
    querystring = {'state_code': state_code, 'offset': offset, 'city': city, 'limit': limit, }
    if price_min:
        querystring['price_min'] = price_min
    if no_hoa_fee:
        querystring['no_hoa_fee'] = no_hoa_fee
    if sort:
        querystring['sort'] = sort
    if property_type:
        querystring['property_type'] = property_type
    if has_3d_tours:
        querystring['has_3d_tours'] = has_3d_tours
    if heating_cooling:
        querystring['heating_cooling'] = heating_cooling
    if open_house:
        querystring['open_house'] = open_house
    if beds_min:
        querystring['beds_min'] = beds_min
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if property_type_nyc_only:
        querystring['property_type_nyc_only'] = property_type_nyc_only
    if baths_min:
        querystring['baths_min'] = baths_min
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    if garage:
        querystring['garage'] = garage
    if lot_views:
        querystring['lot_views'] = lot_views
    if home_age_max:
        querystring['home_age_max'] = home_age_max
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if inside_rooms:
        querystring['inside_rooms'] = inside_rooms
    if days_on_realtor:
        querystring['days_on_realtor'] = days_on_realtor
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if features_in_nyc_only:
        querystring['features_in_nyc_only'] = features_in_nyc_only
    if stories:
        querystring['stories'] = stories
    if price_max:
        querystring['price_max'] = price_max
    if hide_pending_contingent:
        querystring['hide_pending_contingent'] = hide_pending_contingent
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if has_virtual_tours:
        querystring['has_virtual_tours'] = has_virtual_tours
    if hide_foreclosure:
        querystring['hide_foreclosure'] = hide_foreclosure
    if location:
        querystring['location'] = location
    if price_reduced:
        querystring['price_reduced'] = price_reduced
    if baths_max:
        querystring['baths_max'] = baths_max
    if beds_max:
        querystring['beds_max'] = beds_max
    if new_construction:
        querystring['new_construction'] = new_construction
    if outside_features:
        querystring['outside_features'] = outside_features
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if keywords:
        querystring['keywords'] = keywords
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_for_sale(state_code: str, price_min: str=None, price_max: str=None, location: str=None, offset: int=0, city: str='Detroit', limit: int=42, baths_min: str=None, sort: str='newest', hide_pending_contingent: str=None, new_construction: str=None, price_reduced: str=None, beds_max: str=None, lot_size_min: str=None, property_type: str=None, beds_min: str=None, has_3d_tours: str=None, property_type_nyc_only: str=None, baths_max: str=None, home_age_max: str=None, hoa_max: str=None, community_ammenities: str=None, home_size_min: str=None, lot_size_max: str=None, open_house: str=None, days_on_realtor: str=None, include_nearby_areas_slug_id: str=None, no_hoa_fee: str=None, outside_features: str=None, hide_foreclosure: str=None, lot_views: str=None, expand_search_radius: str=None, home_size_max: str=None, garage: str=None, inside_rooms: str=None, heating_cooling: str=None, stories: str=None, features_in_nyc_only: str=None, has_virtual_tours: str=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for-sale properties.
		**Parameters**: ` **state_code**,city, location, sort, limit, offset, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, property_type_nyc_only, new_construction, hide_pending_contingent, has_virtual_tours, has_3d_tours, hide_foreclosure, price_reduced, open_house, keywords, no_hoa_fee, hoa_max, days_on_realtor, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, lot_size_min, lot_size_max, home_age_max, stories, garage, heating_cooling, inside_rooms, outside_features, lot_views, community_ammenities, features_in_nyc_only`"
    state_code: State Code. Get from `/location/suggest` response
        price_min: Minimum list price in USD
        price_max: Maximum list price in USD
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        offset: Offset results, default 0. Maximum 9800.
        city: City name. Get data from `/location/suggest` response
        limit: Number of results. Maximum 200 for Paid Plan, default 42
        baths_min: Minimum bathrooms
        sort: One of the following options: `relevant|newest|lowest_price|highest_price|open_house_date|price_reduced_date|largest_sqft|lot_size|sold_date`. Default is newest
        hide_pending_contingent: `true` for hide pending/contingent. Leave blank for any
        new_construction: `true` for New construction only. Leave blank for any
        price_reduced: `true` for properties with price reduced only. Leave blank for any
        beds_max: Maximum bedrooms
        lot_size_min: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Minimum lot size in sqft
        property_type: Comma separated values. One or more from following options: `multi_family|single_family|mobile|land|farm`
        beds_min: Minimum bedrooms
        has_3d_tours: `true` for properties with 3D tour only. Leave blank for any
        property_type_nyc_only: Comma separated values. One or more from following options: `condo|coop|condop`. For NYC listings only
        baths_max: Maximum bathrooms
        home_age_max: Maximum home age
        hoa_max: Maximum HOA fee in USD
        community_ammenities: Comma separated values. One or more from following options: `community_swimming_pool|community_spa_or_hot_tub|community_golf|community_security_features|community_boat_facilities|tennis_court|community_clubhouse|senior_community`
        home_size_min: One of the following options: `750|1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500`. Minimum home size in sqft
        lot_size_max: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Maximum  lot size in sqft
        open_house: `true` for properties with open house only. Leave blank for any
        days_on_realtor: One of the following options: `today|7|14|21|30`
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-sale-nearby-areas`
        no_hoa_fee: `true` for properties without HOA fee only. Leave blank for any
        outside_features: Comma separated values. One or more from following options: `swimming_pool|spa_or_hot_tub|horse_facilities`
        hide_foreclosure: `true` for hide foreclosure. Leave blank for any
        lot_views: Comma separated values. One or more from following options: `waterfront|cul_de_sac|corner_lot|golf_course_lot_or_frontage|hill_or_mountain_view|ocean_view|lake_view|river_view`
        expand_search_radius: One of the following options: `1|5|10|25|50`. Expand search by radius in miles
        home_size_max: One of the following options: `1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500|10000`. Maximum home size in sqft
        garage: One of the following options: `1+|2+|3+`
        inside_rooms: Comma separated values. One or more comma separated from following options: `basement|hardwood_floors|fireplace|disability_features|den_or_office|family_room|dining_room`
        heating_cooling: Comma separated values. One or more from following options: `central_air|central_heat|forced_air`
        stories: One of the following options: `single|multi`
        features_in_nyc_only: Comma separated values. One or more from following options: `furniture|dishwasher|community_doorman|pets_allowed|laundry_room|elevator|community_outdoor_space`
        has_virtual_tours: `true` for properties with virtual tour only. Leave blank for any
        keywords: Comma separated values. Get popular keywords from `/keywords-search-suggest` response
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v3/for-sale"
    querystring = {'state_code': state_code, }
    if price_min:
        querystring['price_min'] = price_min
    if price_max:
        querystring['price_max'] = price_max
    if location:
        querystring['location'] = location
    if offset:
        querystring['offset'] = offset
    if city:
        querystring['city'] = city
    if limit:
        querystring['limit'] = limit
    if baths_min:
        querystring['baths_min'] = baths_min
    if sort:
        querystring['sort'] = sort
    if hide_pending_contingent:
        querystring['hide_pending_contingent'] = hide_pending_contingent
    if new_construction:
        querystring['new_construction'] = new_construction
    if price_reduced:
        querystring['price_reduced'] = price_reduced
    if beds_max:
        querystring['beds_max'] = beds_max
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if property_type:
        querystring['property_type'] = property_type
    if beds_min:
        querystring['beds_min'] = beds_min
    if has_3d_tours:
        querystring['has_3d_tours'] = has_3d_tours
    if property_type_nyc_only:
        querystring['property_type_nyc_only'] = property_type_nyc_only
    if baths_max:
        querystring['baths_max'] = baths_max
    if home_age_max:
        querystring['home_age_max'] = home_age_max
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    if open_house:
        querystring['open_house'] = open_house
    if days_on_realtor:
        querystring['days_on_realtor'] = days_on_realtor
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if no_hoa_fee:
        querystring['no_hoa_fee'] = no_hoa_fee
    if outside_features:
        querystring['outside_features'] = outside_features
    if hide_foreclosure:
        querystring['hide_foreclosure'] = hide_foreclosure
    if lot_views:
        querystring['lot_views'] = lot_views
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if garage:
        querystring['garage'] = garage
    if inside_rooms:
        querystring['inside_rooms'] = inside_rooms
    if heating_cooling:
        querystring['heating_cooling'] = heating_cooling
    if stories:
        querystring['stories'] = stories
    if features_in_nyc_only:
        querystring['features_in_nyc_only'] = features_in_nyc_only
    if has_virtual_tours:
        querystring['has_virtual_tours'] = has_virtual_tours
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_sale_deprecated(limit: int, state_code: str, city: str, offset: int, stories: str=None, hoa_max: str=None, property_type: str=None, no_hoa_fee: str=None, beds_max: str=None, location: str=None, beds_min: str=None, price_max: str=None, baths_min: str=None, sort: str='newest', price_min: str=None, home_size_max: str=None, baths_max: str=None, property_type_nyc_only: str=None, new_construction: str=None, hide_foreclosure: str=None, days_on_realtor: str=None, open_house: str=None, price_reduced: str=None, include_nearby_areas_slug_id: str=None, keywords: str=None, hide_pending_contingent: str=None, lot_size_max: str=None, has_virtual_tours: str=None, home_age_max: str=None, lot_size_min: str=None, home_size_min: str=None, heating_cooling: str=None, has_3d_tours: str=None, inside_rooms: str=None, expand_search_radius: str=None, outside_features: str=None, garage: str=None, community_ammenities: str=None, lot_views: str=None, features_in_nyc_only: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for-sale properties.
		**Parameters**: `city, state_code, location, limit, offset, sort:newest price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, property_type_nyc_only, new_construction, hide_pending_contingent, has_virtual_tours, has_3d_tours, hide_foreclosure, price_reduced, open_house, keywords, no_hoa_fee, hoa_max, days_on_realtor, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, lot_size_min, lot_size_max, home_age_max, stories, garage, heating_cooling, inside_rooms, outside_features, lot_views, community_ammenities, features_in_nyc_only`"
    limit: Number of results. Maximum 200 for Paid Plan, default 42
        state_code: State Code. Get from `/location/suggest` response
        city: City name. Get data from `/location/suggest` response
        offset: Offset results, default 0. Maximum 9800.
        stories: One of the following options: `single|multi`
        hoa_max: Maximum HOA fee in USD
        property_type: Comma separated values. One or more from following options: `multi_family|single_family|mobile|land|farm`
        no_hoa_fee: `true` for properties without HOA fee only. Leave blank for any
        beds_max: Maximum bedrooms
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        beds_min: Minimum bedrooms
        price_max: Maximum list price in USD
        baths_min: Minimum bathrooms
        sort: One of the following options: `relevant|newest|lowest_price|highest_price|open_house_date|price_reduced_date|largest_sqft|lot_size|sold_date`. Default is relevant
        price_min: Minimum list price in USD
        home_size_max: One of the following options: `1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500|10000`. Maximum home size in sqft
        baths_max: Maximum bathrooms
        property_type_nyc_only: Comma separated values. One or more from following options: `condo|coop|condop`. For NYC listings only
        new_construction: `true` for New construction only. Leave blank for any
        hide_foreclosure: `true` for hide foreclosure. Leave blank for any
        days_on_realtor: One of the following options: `today|7|14|21|30`
        open_house: `true` for properties with open house only. Leave blank for any
        price_reduced: `true` for properties with price reduced only. Leave blank for any
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-sale-nearby-areas`
        keywords: Comma separated values. Get popular keywords from `/keywords-search-suggest` response
        hide_pending_contingent: `true` for hide pending/contingent. Leave blank for any
        lot_size_max: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Maximum  lot size in sqft
        has_virtual_tours: `true` for properties with virtual tour only. Leave blank for any
        home_age_max: Maximum home age
        lot_size_min: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Minimum lot size in sqft
        home_size_min: One of the following options: `750|1000|1250|1500|1750|2000|2250|2500|2750|3000|3250|3500|3750|5000|7500`. Minimum home size in sqft
        heating_cooling: Comma separated values. One or more from following options: `central_air|central_heat|forced_air`
        has_3d_tours: `true` for properties with 3D tour only. Leave blank for any
        inside_rooms: Comma separated values. One or more comma separated from following options: `basement|hardwood_floors|fireplace|disability_features|den_or_office|family_room|dining_room`
        expand_search_radius: One of the following options: `1|5|10|25|50`. Expand search by radius in miles
        outside_features: Comma separated values. One or more from following options: `swimming_pool|spa_or_hot_tub|horse_facilities`
        garage: One of the following options: `1+|2+|3+`
        community_ammenities: Comma separated values. One or more from following options: `community_swimming_pool|community_spa_or_hot_tub|community_golf|community_security_features|community_boat_facilities|tennis_court|community_clubhouse|senior_community`
        lot_views: Comma separated values. One or more from following options: `waterfront|cul_de_sac|corner_lot|golf_course_lot_or_frontage|hill_or_mountain_view|ocean_view|lake_view|river_view`
        features_in_nyc_only: Comma separated values. One or more from following options: `furniture|dishwasher|community_doorman|pets_allowed|laundry_room|elevator|community_outdoor_space`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/for-sale"
    querystring = {'limit': limit, 'state_code': state_code, 'city': city, 'offset': offset, }
    if stories:
        querystring['stories'] = stories
    if hoa_max:
        querystring['hoa_max'] = hoa_max
    if property_type:
        querystring['property_type'] = property_type
    if no_hoa_fee:
        querystring['no_hoa_fee'] = no_hoa_fee
    if beds_max:
        querystring['beds_max'] = beds_max
    if location:
        querystring['location'] = location
    if beds_min:
        querystring['beds_min'] = beds_min
    if price_max:
        querystring['price_max'] = price_max
    if baths_min:
        querystring['baths_min'] = baths_min
    if sort:
        querystring['sort'] = sort
    if price_min:
        querystring['price_min'] = price_min
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if baths_max:
        querystring['baths_max'] = baths_max
    if property_type_nyc_only:
        querystring['property_type_nyc_only'] = property_type_nyc_only
    if new_construction:
        querystring['new_construction'] = new_construction
    if hide_foreclosure:
        querystring['hide_foreclosure'] = hide_foreclosure
    if days_on_realtor:
        querystring['days_on_realtor'] = days_on_realtor
    if open_house:
        querystring['open_house'] = open_house
    if price_reduced:
        querystring['price_reduced'] = price_reduced
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if keywords:
        querystring['keywords'] = keywords
    if hide_pending_contingent:
        querystring['hide_pending_contingent'] = hide_pending_contingent
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    if has_virtual_tours:
        querystring['has_virtual_tours'] = has_virtual_tours
    if home_age_max:
        querystring['home_age_max'] = home_age_max
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if heating_cooling:
        querystring['heating_cooling'] = heating_cooling
    if has_3d_tours:
        querystring['has_3d_tours'] = has_3d_tours
    if inside_rooms:
        querystring['inside_rooms'] = inside_rooms
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if outside_features:
        querystring['outside_features'] = outside_features
    if garage:
        querystring['garage'] = garage
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if lot_views:
        querystring['lot_views'] = lot_views
    if features_in_nyc_only:
        querystring['features_in_nyc_only'] = features_in_nyc_only
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_rent_similiar_homes_deprecated(property_id: int, postal_code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similiar for-rent homes by `property_id`"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/for-rent/similiar-homes"
    querystring = {'property_id': property_id, 'postal_code': postal_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_rent_deprecated(city: str, state_code: str, sort: str='lowest_price', home_size_max: int=3000, offset: int=0, beds_min: int=1, price_min: int=1000, limit: int=10, location: int=48278, baths_max: int=5, expand_search_radius: int=25, baths_min: int=1, price_max: int=3000, cats_ok: bool=None, home_size_min: int=500, include_nearby_areas_slug_id: str='Union-City_NJ,Howard-Beach_NY', in_unit_features: str='central_air', community_ammenities: str='garage_1_or_more', dogs_ok: bool=None, beds_max: int=5, property_type: str='apartment', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get for-rent properties.
		**Parameters**: `city, state_code, location, limit, offset, sort, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, in_unit_features, community_ammenities, cats_ok, dogs_ok`"
    city: City name. Get data from `/location/suggest` response
        state_code: State Code. Get from `/location/suggest` response
        sort: One of the following options: `frehsnest|recently_added_update|lowest_price|highest_price`. Default is frehsnest
        home_size_max: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        offset: Offset results, default 0. Maximum 9800.
        beds_min: Minimum bedrooms
        price_min: Minimum list price in USD
        limit: Number of results. Maximum 200 for Paid Plan, default 42
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        baths_max: Maximum bathrooms
        expand_search_radius: One of the following options: `1|5|10|25|50`
        baths_min: Minimum bathrooms
        price_max: Maximum list price in USD
        cats_ok: `true` for Cats allowed only
        home_size_min: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-rent-nearby-areas`
        in_unit_features: Comma separated values. One or more from following options: `central_air|dishwasher|washer_dryer|furnished`
        community_ammenities: Comma separated values. One or more from following options: `garage_1_or_more|swimming_pool|community_doorman|community_outdoor_space|community_elevator|laundry_room|community_gym`
        dogs_ok: `true` for Dogs allowed only
        beds_max: Maximum bedrooms
        property_type: Comma separated values. One or more from following options: `townhome,coop,single_family,apartment,condo,condop`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/for-rent"
    querystring = {'city': city, 'state_code': state_code, }
    if sort:
        querystring['sort'] = sort
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if offset:
        querystring['offset'] = offset
    if beds_min:
        querystring['beds_min'] = beds_min
    if price_min:
        querystring['price_min'] = price_min
    if limit:
        querystring['limit'] = limit
    if location:
        querystring['location'] = location
    if baths_max:
        querystring['baths_max'] = baths_max
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if baths_min:
        querystring['baths_min'] = baths_min
    if price_max:
        querystring['price_max'] = price_max
    if cats_ok:
        querystring['cats_ok'] = cats_ok
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if in_unit_features:
        querystring['in_unit_features'] = in_unit_features
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if dogs_ok:
        querystring['dogs_ok'] = dogs_ok
    if beds_max:
        querystring['beds_max'] = beds_max
    if property_type:
        querystring['property_type'] = property_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_for_rent_result_count(city: str, state_code: str, beds_min: int=1, dogs_ok: bool=None, include_nearby_areas_slug_id: str='Union-City_NJ,Howard-Beach_NY', price_min: int=1000, beds_max: int=5, baths_min: int=1, cats_ok: bool=None, price_max: int=3000, location: int=48278, in_unit_features: str='central_air', home_size_min: int=500, home_size_max: int=3000, community_ammenities: str='garage_1_or_more', expand_search_radius: int=25, property_type: str='apartment', baths_max: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get result count for-rent properties.
		**Parameters**: `city, state_code, location, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, in_unit_features, community_ammenities, cats_ok, dogs_ok`"
    city: City name. Get data from `/location/suggest` response
        state_code: State Code. Get from `/location/suggest` response
        beds_min: Minimum bedrooms
        dogs_ok: `true` for Dogs allowed only
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-rent-nearby-areas`
        price_min: Minimum list price in USD
        beds_max: Maximum bedrooms
        baths_min: Minimum bathrooms
        cats_ok: `true` for Cats allowed only
        price_max: Maximum list price in USD
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        in_unit_features: Comma separated values. One or more from following options: `central_air|dishwasher|washer_dryer|furnished`
        home_size_min: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        home_size_max: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        community_ammenities: Comma separated values. One or more from following options: `garage_1_or_more|swimming_pool|community_doorman|community_outdoor_space|community_elevator|laundry_room|community_gym`
        expand_search_radius: One of the following options: `1|5|10|25|50`
        property_type: Comma separated values. One or more from following options: `townhome,coop,single_family,apartment,condo,condop`
        baths_max: Maximum bathrooms
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/for-rent-result-count"
    querystring = {'city': city, 'state_code': state_code, }
    if beds_min:
        querystring['beds_min'] = beds_min
    if dogs_ok:
        querystring['dogs_ok'] = dogs_ok
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if price_min:
        querystring['price_min'] = price_min
    if beds_max:
        querystring['beds_max'] = beds_max
    if baths_min:
        querystring['baths_min'] = baths_min
    if cats_ok:
        querystring['cats_ok'] = cats_ok
    if price_max:
        querystring['price_max'] = price_max
    if location:
        querystring['location'] = location
    if in_unit_features:
        querystring['in_unit_features'] = in_unit_features
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if property_type:
        querystring['property_type'] = property_type
    if baths_max:
        querystring['baths_max'] = baths_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_for_rent_by_zipcode(zipcode: int, sort: str='lowest_price', offset: int=0, beds_min: int=None, home_size_max: int=None, baths_max: int=None, limit: int=10, price_min: int=None, community_ammenities: str=None, in_unit_features: str=None, include_nearby_areas_slug_id: str=None, home_size_min: int=None, dogs_ok: bool=None, baths_min: int=None, property_type: str=None, cats_ok: bool=None, beds_max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get for-rent properties.
		**Parameters**: `zipcode, limit, offset, sort, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, in_unit_features, community_ammenities, cats_ok, dogs_ok`"
    zipcode: zipcode
        sort: One of the following options: `frehsnest|recently_added_update|lowest_price|highest_price`. Default is frehsnest
        offset: Offset results, default 0. Maximum 9800.
        beds_min: Minimum bedrooms
        home_size_max: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        baths_max: Maximum bathrooms
        limit: Number of results. Maximum 200 for Paid Plan, default 42
        price_min: Minimum list price in USD
        community_ammenities: Comma separated values. One or more from following options: `garage_1_or_more|swimming_pool|community_doorman|community_outdoor_space|community_elevator|laundry_room|community_gym`
        in_unit_features: Comma separated values. One or more from following options: `central_air|dishwasher|washer_dryer|furnished`
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-rent-nearby-areas`
        home_size_min: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        dogs_ok: `true` for Dogs allowed only
        baths_min: Minimum bathrooms
        property_type: Comma separated values. One or more from following options: `townhome,coop,single_family,apartment,condo,condop`
        cats_ok: `true` for Cats allowed only
        beds_max: Maximum bedrooms
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/for-rent-by-zipcode"
    querystring = {'zipcode': zipcode, }
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if beds_min:
        querystring['beds_min'] = beds_min
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if baths_max:
        querystring['baths_max'] = baths_max
    if limit:
        querystring['limit'] = limit
    if price_min:
        querystring['price_min'] = price_min
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if in_unit_features:
        querystring['in_unit_features'] = in_unit_features
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if dogs_ok:
        querystring['dogs_ok'] = dogs_ok
    if baths_min:
        querystring['baths_min'] = baths_min
    if property_type:
        querystring['property_type'] = property_type
    if cats_ok:
        querystring['cats_ok'] = cats_ok
    if beds_max:
        querystring['beds_max'] = beds_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_for_rent(state_code: str, city: str, expand_search_radius: int=25, price_max: int=3000, offset: int=0, dogs_ok: bool=None, home_size_max: int=3000, sort: str='lowest_price', baths_max: int=5, beds_max: int=5, cats_ok: bool=None, limit: int=10, home_size_min: int=500, property_type: str='apartment', location: int=48278, beds_min: int=1, include_nearby_areas_slug_id: str='Union-City_NJ,Howard-Beach_NY', baths_min: int=1, price_min: int=1000, community_ammenities: str='garage_1_or_more', in_unit_features: str='central_air', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get for-rent properties.
		**Parameters**: `city, state_code, location, limit, offset, sort, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, in_unit_features, community_ammenities, cats_ok, dogs_ok`"
    state_code: State Code. Get from `/location/suggest` response
        city: City name. Get data from `/location/suggest` response
        expand_search_radius: One of the following options: `1|5|10|25|50`
        price_max: Maximum list price in USD
        offset: Offset results, default 0. Maximum 9800.
        dogs_ok: `true` for Dogs allowed only
        home_size_max: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        sort: One of the following options: `frehsnest|recently_added_update|lowest_price|highest_price`. Default is frehsnest
        baths_max: Maximum bathrooms
        beds_max: Maximum bedrooms
        cats_ok: `true` for Cats allowed only
        limit: Number of results. Maximum 200 for Paid Plan, default 42
        home_size_min: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        property_type: Comma separated values. One or more from following options: `townhome,coop,single_family,apartment,condo,condop`
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        beds_min: Minimum bedrooms
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-rent-nearby-areas`
        baths_min: Minimum bathrooms
        price_min: Minimum list price in USD
        community_ammenities: Comma separated values. One or more from following options: `garage_1_or_more|swimming_pool|community_doorman|community_outdoor_space|community_elevator|laundry_room|community_gym`
        in_unit_features: Comma separated values. One or more from following options: `central_air|dishwasher|washer_dryer|furnished`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/for-rent"
    querystring = {'state_code': state_code, 'city': city, }
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if price_max:
        querystring['price_max'] = price_max
    if offset:
        querystring['offset'] = offset
    if dogs_ok:
        querystring['dogs_ok'] = dogs_ok
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if sort:
        querystring['sort'] = sort
    if baths_max:
        querystring['baths_max'] = baths_max
    if beds_max:
        querystring['beds_max'] = beds_max
    if cats_ok:
        querystring['cats_ok'] = cats_ok
    if limit:
        querystring['limit'] = limit
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if property_type:
        querystring['property_type'] = property_type
    if location:
        querystring['location'] = location
    if beds_min:
        querystring['beds_min'] = beds_min
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if baths_min:
        querystring['baths_min'] = baths_min
    if price_min:
        querystring['price_min'] = price_min
    if community_ammenities:
        querystring['community_ammenities'] = community_ammenities
    if in_unit_features:
        querystring['in_unit_features'] = in_unit_features
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_for_rent_similiar_homes(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similiar for-rent homes by `property_id`"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/for-rent/similiar-homes"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_agents_search(city: str='Manhattan', offset: int=None, recommendations_count_min: int=None, price_min: int=None, limit: int=None, postal_code: str=None, state_code: str='NY', sort: str=None, agent_name: str=None, price_max: int=None, types: str=None, agent_rating_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for agents, teams and office"
    city: City name. Required if not search by postal_code.
        offset: Offset. Default is 0
        recommendations_count_min: Minimum recommendations count. `1 to 10`. Default is `Any`
        price_min: Minimum list price in USD
        limit: Maximum is 100
        postal_code: Postal code. Required if search by postal_code only.
        state_code: State code. Required if not search by postal_code.
        sort: One of the following options: `agent_rating_high|recent_activity_high|recommendations_count_high|for_sale_count_high|recently_sold_high`
        agent_name: Agent name to search.
        price_max: Maximum list price in USD
        types: One of the following options: `agent | team | office`
        agent_rating_min: Minimum agent rating. `1 to 5`. Default is `Any`.
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/agents/agents-search"
    querystring = {}
    if city:
        querystring['city'] = city
    if offset:
        querystring['offset'] = offset
    if recommendations_count_min:
        querystring['recommendations_count_min'] = recommendations_count_min
    if price_min:
        querystring['price_min'] = price_min
    if limit:
        querystring['limit'] = limit
    if postal_code:
        querystring['postal_code'] = postal_code
    if state_code:
        querystring['state_code'] = state_code
    if sort:
        querystring['sort'] = sort
    if agent_name:
        querystring['agent_name'] = agent_name
    if price_max:
        querystring['price_max'] = price_max
    if types:
        querystring['types'] = types
    if agent_rating_min:
        querystring['agent_rating_min'] = agent_rating_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_agent_profile(advertiser_id: str, nrds_id: str='647587948', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Agent's profile by advertiser_id and nrds_id"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/agents/agent-profile"
    querystring = {'advertiser_id': advertiser_id, }
    if nrds_id:
        querystring['nrds_id'] = nrds_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_agent_listings(advertiser_id: str, nrds_id: str='476508717', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Agent's listings"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/agents/agent-listings"
    querystring = {'advertiser_id': advertiser_id, }
    if nrds_id:
        querystring['nrds_id'] = nrds_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_agents_search_by_zipcode(zipcode: str, sort: str=None, offset: int=None, limit: int=None, recommendations_count_min: int=None, types: str=None, agent_rating_min: int=None, price_max: int=None, agent_name: str=None, price_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for agents, teams, and office by zip code"
    zipcode: Postal code. Required if search by postal_code only.
        sort: One of the following options: `agent_rating_high|recent_activity_high|recommendations_count_high|for_sale_count_high|recently_sold_high`
        offset: Offset. Default is 0
        limit: Maximum is 20
        recommendations_count_min: Minimum recommendations count. `1 to 10`. Default is `Any`
        types: One of the following options: `agent | team | office`
        agent_rating_min: Minimum agent rating. `1 to 5`. Default is `Any`.
        price_max: Maximum list price in USD
        agent_name: Agent name to search.
        price_min: Minimum list price in USD
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/agents/agents-search-by-zipcode"
    querystring = {'zipcode': zipcode, }
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if recommendations_count_min:
        querystring['recommendations_count_min'] = recommendations_count_min
    if types:
        querystring['types'] = types
    if agent_rating_min:
        querystring['agent_rating_min'] = agent_rating_min
    if price_max:
        querystring['price_max'] = price_max
    if agent_name:
        querystring['agent_name'] = agent_name
    if price_min:
        querystring['price_min'] = price_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_sold_homes_by_zipcode(zipcode: int, sort: str='sold_date', offset: int=0, price_max: int=None, max_sold_days: int=None, beds_max: int=None, beds_min: int=None, baths_min: int=None, baths_max: int=None, property_type: str=None, include_nearby_areas_slug_id: str=None, expand_search_radius: int=None, home_size_max: int=None, lot_size_min: str=None, home_age_max: str=None, home_size_min: int=None, lot_size_max: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for-sale properties. 
		**Parameters**: `zipcode, limit, offset, sort, max_sold_days, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, lot_size_min, lot_size_max, home_age_max`"
    zipcode: zipcode
        sort: One of the following options: `sold_date | lowest_price | highest_price | lot_size | number_of_beds`. Default is sold_date
        offset: Offset results, default 0
        price_max: Maximum list price in USD
        max_sold_days: Maximum sold days form now
        beds_max: Maximum bedrooms
        beds_min: Minimum bedrooms
        baths_min: Minimum bathrooms
        baths_max: Maximum bathrooms
        property_type: Comma separated values. One or more from following options: `multi_family|single_family|mobile|land|farm`
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-rent-nearby-areas`
        expand_search_radius: One of the following options: `1|5|10|25|50`
        home_size_max: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        lot_size_min: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Minimum lot size in sqft
        home_age_max: Maximum home age
        home_size_min: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        lot_size_max: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Maximum  lot size in sqft
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/v2/sold-homes-by-zipcode"
    querystring = {'zipcode': zipcode, }
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if price_max:
        querystring['price_max'] = price_max
    if max_sold_days:
        querystring['max_sold_days'] = max_sold_days
    if beds_max:
        querystring['beds_max'] = beds_max
    if beds_min:
        querystring['beds_min'] = beds_min
    if baths_min:
        querystring['baths_min'] = baths_min
    if baths_max:
        querystring['baths_max'] = baths_max
    if property_type:
        querystring['property_type'] = property_type
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if home_age_max:
        querystring['home_age_max'] = home_age_max
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finance_mortgage_calculate(show_amortization: bool, hoa_fees: int, percent_tax_rate: int, year_term: int, percent_rate: int, down_payment: int, monthly_home_insurance: int, price: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mortgage calculae"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/finance/mortgage-calculate"
    querystring = {'show_amortization': show_amortization, 'hoa_fees': hoa_fees, 'percent_tax_rate': percent_tax_rate, 'year_term': year_term, 'percent_rate': percent_rate, 'down_payment': down_payment, 'monthly_home_insurance': monthly_home_insurance, 'price': price, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finance_rate_trends(is_refinance: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current rate trends and historical rate trends"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/finance/rate-trends"
    querystring = {'is_refinance': is_refinance, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finance_average_rate(postal_code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get average rates data"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/finance/average-rate"
    querystring = {'postal_code': postal_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sold_homes(city: str, state_code: str, price_min: int=None, limit: int=10, location: int=None, beds_min: int=None, baths_min: int=None, beds_max: int=None, include_nearby_areas_slug_id: str=None, home_age_max: str=None, expand_search_radius: int=None, home_size_min: int=None, max_sold_days: int=None, offset: int=0, baths_max: int=None, sort: str='sold_date', price_max: int=None, home_size_max: int=None, property_type: str=None, lot_size_min: str=None, lot_size_max: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for-sale properties. 
		**Parameters**: `city, state_code, location, limit, offset, sort, max_sold_days, price_min, price_max, beds_min, beds_max, baths_min, baths_max, property_type, expand_search_radius, include_nearby_areas_slug_id, home_size_min, home_size_max, lot_size_min, lot_size_max, home_age_max`"
    city: City name. Get data from `/location/suggest` response
        state_code: State Code. Get from `/location/suggest` response
        price_min: Minimum list price in USD
        limit: Number of results. Maximum 200 for Paid Plan, default 42
        location: Additional Location detail, could be neighborhood or postal_code or leave it blank. Get from `/location/suggest` response. Default is blank
        beds_min: Minimum bedrooms
        baths_min: Minimum bathrooms
        beds_max: Maximum bedrooms
        include_nearby_areas_slug_id: Comma separated values. Expand search by including nearby areas. Get slug_id from `/location/for-rent-nearby-areas`
        home_age_max: Maximum home age
        expand_search_radius: One of the following options: `1|5|10|25|50`
        home_size_min: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        max_sold_days: Maximum sold days form now
        offset: Offset results, default 0
        baths_max: Maximum bathrooms
        sort: One of the following options: `sold_date | lowest_price | highest_price | lot_size | number_of_beds`. Default is sold_date
        price_max: Maximum list price in USD
        home_size_max: One of the following options: `500|750|1000|1250|1500|1750|2000|2250|2500|2750|3000`
        property_type: Comma separated values. One or more from following options: `multi_family|single_family|mobile|land|farm`
        lot_size_min: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Minimum lot size in sqft
        lot_size_max: One of the following options: `2000|300|4000|5000|7500|10890|21780|43560|87120|217800|435600|653400|871200`. Maximum  lot size in sqft
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/sold-homes"
    querystring = {'city': city, 'state_code': state_code, }
    if price_min:
        querystring['price_min'] = price_min
    if limit:
        querystring['limit'] = limit
    if location:
        querystring['location'] = location
    if beds_min:
        querystring['beds_min'] = beds_min
    if baths_min:
        querystring['baths_min'] = baths_min
    if beds_max:
        querystring['beds_max'] = beds_max
    if include_nearby_areas_slug_id:
        querystring['include_nearby_areas_slug_id'] = include_nearby_areas_slug_id
    if home_age_max:
        querystring['home_age_max'] = home_age_max
    if expand_search_radius:
        querystring['expand_search_radius'] = expand_search_radius
    if home_size_min:
        querystring['home_size_min'] = home_size_min
    if max_sold_days:
        querystring['max_sold_days'] = max_sold_days
    if offset:
        querystring['offset'] = offset
    if baths_max:
        querystring['baths_max'] = baths_max
    if sort:
        querystring['sort'] = sort
    if price_max:
        querystring['price_max'] = price_max
    if home_size_max:
        querystring['home_size_max'] = home_size_max
    if property_type:
        querystring['property_type'] = property_type
    if lot_size_min:
        querystring['lot_size_min'] = lot_size_min
    if lot_size_max:
        querystring['lot_size_max'] = lot_size_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_noise_score(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location noise score by (**latitude & longitude**)"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/location/noise-score"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_for_rent_nearby_areas(area_type: str, city: str=None, neighborhood: str=None, postal_code: int=14218, state_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearby areas for **include_nearby_areas_slug_id** parameter in **/for-rent**. Get by (area_type="city" & city & state_code) or by (area_type="neighborhood" & city & state_code & neighborhood) or by (area_type="postal_code" & postal_code)"
    area_type: One of the following options: `city|postal_code|neighborhood`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/location/for-rent-nearby-areas"
    querystring = {'area_type': area_type, }
    if city:
        querystring['city'] = city
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    if postal_code:
        querystring['postal_code'] = postal_code
    if state_code:
        querystring['state_code'] = state_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_for_sale_nearby_areas_by_postal_code(postal_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearby areas by `postal_code` for **include_nearby_areas_slug_id** parameter in **/for-sale** endpoint."
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/location/for-sale-nearby-areas-by-postal-code"
    querystring = {'postal_code': postal_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_for_sale_nearby_areas(area_type: str, postal_code: str=None, state_code: str='NY', neighborhood: str=None, city: str='New York', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearby areas for **include_nearby_areas_slug_id** parameter in **/for-sale** endpoint. Get by (area_type="city" & city & state_code) or by (area_type="neighborhood" & city & state_code & neighborhood) or by (area_type="postal_code" & postal_code)"
    area_type: One of the following options: `city|neighborhood`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/location/for-sale-nearby-areas"
    querystring = {'area_type': area_type, }
    if postal_code:
        querystring['postal_code'] = postal_code
    if state_code:
        querystring['state_code'] = state_code
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_suggest(input: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location suggestion / autocomplete
		**Required Parameter**: `input`
		**Optional Parameter**:"
    input: Part of location name
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/location/suggest"
    querystring = {'input': input, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_schools(state_code: str=None, neighborhood: str=None, city: str=None, postal_code: int=14218, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get schools near a location by (**state_code & city**) or by (**state_code & city & neighborhood**) or by **postal_code**"
    
    """
    url = f"https://us-real-estate.p.rapidapi.com/location/schools"
    querystring = {}
    if state_code:
        querystring['state_code'] = state_code
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    if city:
        querystring['city'] = city
    if postal_code:
        querystring['postal_code'] = postal_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_commute_time(mode: str, origins: str, destinations: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get commute time from origins to destinations with one of following mode: walking|driving|bicycling|transit"
    mode: One of the following options: `driving|walking|bicycling|transit`
        origins: Origin location: `address, city+state_code, neighborhood, postal_code, etc`
        destinations: Destination location: `address, city+state_code, neighborhood, postal_code, etc`
        
    """
    url = f"https://us-real-estate.p.rapidapi.com/location/commute-time"
    querystring = {'mode': mode, 'origins': origins, 'destinations': destinations, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-real-estate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

