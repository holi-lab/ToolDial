import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mortgage_calculate_deprecated(rate: int, price: int, hoi: int, downpayment: int, tax_rate: int, term: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate mortgage"
    rate: Interest rate
        price: Home price
        hoi: Home insurance
        downpayment: Down payment
        tax_rate: Property tax
        term: Mortgage loan type
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/mortgage/calculate"
    querystring = {'rate': rate, 'price': price, 'hoi': hoi, 'downpayment': downpayment, 'tax_rate': tax_rate, 'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mortgage_check_rates_deprecated(propertyprice: int, loanpercent: int, zip: str, loanpurpose: str, loantypes: str, points: str, creditscore: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check mortgage rates"
    zip: The postal code
        loanpurpose: purchase|refinance
        loantypes: One of the following : ALL|30YrFixed|30YrFixed&#95;FHA|30YrFixed&#95;VA|20YrFixed|15YrFixed|10YrFixed|5Arm|7Arm
        points: One of the following : all|zero|zero&#95;to&#95;one|one&#95;to&#95;two
        creditscore: One of the following : excellent|good|fair|poor
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/mortgage/check-rates"
    querystring = {'propertyPrice': propertyprice, 'loanPercent': loanpercent, 'zip': zip, 'loanPurpose': loanpurpose, 'loanTypes': loantypes, 'points': points, 'creditScore': creditscore, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_list_for_sale_deprecated(offset: int, state_code: str, city: str, limit: int, sort: str='relevance', beds_min: int=None, radius: int=None, is_foreclosure: bool=None, age_min: int=None, is_new_plan: bool=None, lot_sqft_min: int=None, price_max: int=None, is_matterports: bool=None, lot_sqft_max: int=None, lat_min: int=None, lng_max: int=None, price_min: int=None, baths_min: int=None, is_new_construction: bool=None, prop_sub_type: str=None, prop_type: str=None, age_max: int=None, sqft_max: int=None, sqft_min: int=None, is_pending: bool=None, features: str=None, postal_code: str=None, is_contingent: bool=None, has_open_house: bool=None, lng_min: int=None, lat_max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties for sale"
    offset: The offset of items to be ignored in response for paging
        state_code: The value of state&#95;code field responded in locations/auto-complete API (do not use this parameter with postal&#95;code)
        city: The value of city field responded in locations/auto-complete API (do not use this parameter with postal_code)
        limit: The number of items to be responded in every request
        sort: One of the followings (separated by comma for multiple values): relevance|newest|price&#95;low|price&#95;high|photos|open&#95;house&#95;date|sqft&#95;high|price&#95;reduced&#95;date
        beds_min: Min beds of properties
        radius: Radius in miles to look for properties (1 to 20)
        is_foreclosure: true/false - Foreclosures only
        age_min: Min age of properties
        is_new_plan: true/false - Homes not yet built
        lot_sqft_min: Min Lot/Acreage size
        price_max: Option filter by setting max price
        is_matterports: true/false - 3D tours only
        lot_sqft_max: Max Lot/Acreage size
        lat_min: Look for properties in bounding box, this is the min latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lng_max: Look for properties in bounding box, this is the max longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        price_min: Option filter by setting min price
        baths_min: Min baths of properties
        is_new_construction: true/false - New construction only
        prop_sub_type: One of the followings (separated by comma for multiple values): condo,cond&#95;op,townhouse,co&#95;op
        prop_type: One of the followings (separated by comma for multiple values): single&#95;family,multi&#95;family,condo,mobile,land,farm,other
        age_max: Max age of properties
        sqft_max: Max size of the properties
        sqft_min: Min size of the properties
        is_pending: true/false - Pending only
        features: One of the followings (separated by comma for multiple values): garage&#95;2&#95;or&#95;more,view,waterfront,golf&#95;course&#95;view,swimming&#95;pool,cul&#95;de&#95;sac,hardwood&#95;floors,basement,fireplace,energy&#95;efficient,disability&#95;features,dining&#95;room,washer&#95;dryer,family&#95;room,den&#95;or&#95;office,game&#95;room,central&#95;air,central&#95;heat,forced&#95;air,single&#95;story,two&#95;or&#95;more&#95;stories,corner&#95;lot,water&#95;view,golf&#95;course&#95;lot&#95;or&#95;frontage,hill&#95;or&#95;mountain&#95;view,ocean&#95;view,city&#95;view,lake&#95;view,river&#95;view,community&#95;security&#95;features,community&#95;swimming&#95;pool,community&#95;boat&#95;facilities,recreation&#95;facilities,community&#95;clubhouse,community&#95;horse&#95;facilities,community&#95;tennis&#95;court,community&#95;park,community&#95;golf,senior&#95;community,community&#95;spa&#95;or&#95;hot&#95;tub,rv&#95;or&#95;boat&#95;parking,horse&#95;facilities,tennis&#95;court,spa&#95;or&#95;hot&#95;tub,garage&#95;1&#95;or&#95;more,garage&#95;3&#95;or&#95;more,carport
        postal_code: Zip code or postal code (do not use this parameter with city and state_code)
        is_contingent: true/false - Contingent only
        has_open_house: true/false - Open houses only
        lng_min: Look for properties in bounding box, this is the min longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lat_max: Look for properties in bounding box, this is the max latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v2/list-for-sale"
    querystring = {'offset': offset, 'state_code': state_code, 'city': city, 'limit': limit, }
    if sort:
        querystring['sort'] = sort
    if beds_min:
        querystring['beds_min'] = beds_min
    if radius:
        querystring['radius'] = radius
    if is_foreclosure:
        querystring['is_foreclosure'] = is_foreclosure
    if age_min:
        querystring['age_min'] = age_min
    if is_new_plan:
        querystring['is_new_plan'] = is_new_plan
    if lot_sqft_min:
        querystring['lot_sqft_min'] = lot_sqft_min
    if price_max:
        querystring['price_max'] = price_max
    if is_matterports:
        querystring['is_matterports'] = is_matterports
    if lot_sqft_max:
        querystring['lot_sqft_max'] = lot_sqft_max
    if lat_min:
        querystring['lat_min'] = lat_min
    if lng_max:
        querystring['lng_max'] = lng_max
    if price_min:
        querystring['price_min'] = price_min
    if baths_min:
        querystring['baths_min'] = baths_min
    if is_new_construction:
        querystring['is_new_construction'] = is_new_construction
    if prop_sub_type:
        querystring['prop_sub_type'] = prop_sub_type
    if prop_type:
        querystring['prop_type'] = prop_type
    if age_max:
        querystring['age_max'] = age_max
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if is_pending:
        querystring['is_pending'] = is_pending
    if features:
        querystring['features'] = features
    if postal_code:
        querystring['postal_code'] = postal_code
    if is_contingent:
        querystring['is_contingent'] = is_contingent
    if has_open_house:
        querystring['has_open_house'] = has_open_house
    if lng_min:
        querystring['lng_min'] = lng_min
    if lat_max:
        querystring['lat_max'] = lat_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_list_by_mls_deprecated(mls_id: str, prop_status: str=None, limit: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties by MLS ID"
    mls_id: The MLS ID
        prop_status: One of the followings : for&#95;sale|for&#95;rent|recently&#95;sold
        limit: The number of items per response
        offset: The offset of items to be ignored in response for paging
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v2/list-by-mls"
    querystring = {'mls_id': mls_id, }
    if prop_status:
        querystring['prop_status'] = prop_status
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_detail_deprecated(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detail information"
    property_id: The value of property_id field returned in  .../properties/list-.... or  .../properties/v2/list-... endpoints.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v2/detail"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_for_rent_deprecated(city: str, offset: int, limit: int, state_code: str, no_pets_allowed: bool=None, sqft_min: int=None, lng_max: int=None, price_min: int=None, sqft_max: int=None, mapi_community_features: str=None, sort: str='relevance', postal_code: str=None, allows_dogs: bool=None, lat_min: int=None, baths_min: int=None, beds_min: int=None, prop_type: str=None, price_max: int=None, lat_max: int=None, radius: int=None, listed_date_min: str=None, allows_cats: bool=None, lng_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties for rent"
    city: The value of city field responded in locations/auto-complete API (do not use this parameter with postal_code)
        offset: The offset of items to be ignored in response for paging
        limit: The number of items to be responded in every request
        state_code: The value of state&#95;code field responded in locations/auto-complete API (do not use this parameter with postal&#95;code)
        no_pets_allowed: true/false - Pets are allowed or not (cannot be used with allows&#95;dogs, allows&#95;cats)
        sqft_min: Min size of the properties
        lng_max: Look for properties in bounding box, this is the max longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        price_min: Option filter by setting min price
        sqft_max: Max size of the properties
        mapi_community_features: One of the followings (separated by comma for multiple values): community&#95;washer&#95;dryer,community&#95;parking,central&#95;air,community&#95;pool,community&#95;gym
        sort: One of the followings : completeness,photos,freshest|price&#95;low|price&#95;high|photos|newest
        postal_code: Zip code or postal code (do not use this parameter with city and state_code)
        allows_dogs: true/false - Dogs are allowed or not (cannot be used with no&#95;pets&#95;allowed)
        lat_min: Look for properties in bounding box, this is the min latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        baths_min: Option filter by setting at least the number of bathrooms
        beds_min: Option filter by setting at least the number of bedrooms
        prop_type: One of the followings (separated by comma for multiple values): apartment,single&#95;family,mapi&#95;condo&#95;townhome,other
        price_max: Option filter by setting max price
        lat_max: Look for properties in bounding box, this is the max latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        radius: Radius in miles to look for properties
        listed_date_min: Date string format yyyy-MM-dd'T'HH:mm:ss'Z' , such as : 2019-08-01T16:24:40Z .The date from which the properties have been on realtor.com
        allows_cats: true/false - Cats are allowed or not (cannot be used with no&#95;pets&#95;allowed)
        lng_min: Look for properties in bounding box, this is the min longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/list-for-rent"
    querystring = {'city': city, 'offset': offset, 'limit': limit, 'state_code': state_code, }
    if no_pets_allowed:
        querystring['no_pets_allowed'] = no_pets_allowed
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if lng_max:
        querystring['lng_max'] = lng_max
    if price_min:
        querystring['price_min'] = price_min
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    if mapi_community_features:
        querystring['mapi_community_features'] = mapi_community_features
    if sort:
        querystring['sort'] = sort
    if postal_code:
        querystring['postal_code'] = postal_code
    if allows_dogs:
        querystring['allows_dogs'] = allows_dogs
    if lat_min:
        querystring['lat_min'] = lat_min
    if baths_min:
        querystring['baths_min'] = baths_min
    if beds_min:
        querystring['beds_min'] = beds_min
    if prop_type:
        querystring['prop_type'] = prop_type
    if price_max:
        querystring['price_max'] = price_max
    if lat_max:
        querystring['lat_max'] = lat_max
    if radius:
        querystring['radius'] = radius
    if listed_date_min:
        querystring['listed_date_min'] = listed_date_min
    if allows_cats:
        querystring['allows_cats'] = allows_cats
    if lng_min:
        querystring['lng_min'] = lng_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_list_for_rent_deprecated(limit: int, city: str, state_code: str, offset: int, price_max: int=None, baths_min: int=None, beds_min: int=None, allows_cats: bool=None, lot_sqft_min: int=None, features: str=None, sqft_max: int=None, lot_sqft_max: int=None, radius: int=None, lat_min: int=None, allows_dogs: bool=None, sqft_min: int=None, price_min: int=None, postal_code: str=None, prop_sub_type: str=None, prop_type: str=None, sort: str='relevance', lng_max: int=None, lat_max: int=None, lng_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties for rent"
    limit: The number of items to be responded in every request
        city: The value of city field responded in locations/auto-complete API (do not use this parameter with postal_code)
        state_code: The value of state&#95;code field responded in locations/auto-complete API (do not use this parameter with postal&#95;code)
        offset: The offset of items to be ignored in response for paging
        price_max: Option filter by setting max price
        baths_min: Min baths of properties
        beds_min: Min beds of `properties`
        allows_cats: false/true - Whether or not cats are allowed to live in properties
        lot_sqft_min: Min Lot/Acreage size
        features: One of the followings (separated by comma for multiple values): recreation&#95;facilities,swimming&#95;pool,washer&#95;dryer,garage&#95;1&#95;or&#95;more,central&#95;air,fireplace,spa&#95;or&#95;hot&#95;tub,dishwasher,community&#95;doorman,community&#95;elevator,furnished,laundry&#95;room,community&#95;no&#95;fee,community&#95;outdoor&#95;space,pets&#95;allowed
        sqft_max: Max size of the properties
        lot_sqft_max: Max Lot/Acreage size
        radius: Radius in miles to look for properties (1 to 20)
        lat_min: Look for properties in bounding box, this is the min latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        allows_dogs: false/true - Whether or not dogs are allowed to live in properties
        sqft_min: Min size of the properties
        price_min: Option filter by setting min price
        postal_code: Zip code or postal code (do not use this parameter with city and state&#95;code)
        prop_sub_type: One of the followings (separated by comma for multiple values): condo,cond&#95;op,townhouse,co&#95;op
        prop_type: One of the followings (separated by comma for multiple values): single&#95;family,multi&#95;family,condo,mobile,land,farm,other
        sort: One of the followings : relevance|newest|price&#95;low|price&#95;high|photos|open&#95;house&#95;date|sqft&#95;high|price&#95;reduced&#95;date
        lng_max: Look for properties in bounding box, this is the max longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lat_max: Look for properties in bounding box, this is the max latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lng_min: Look for properties in bounding box, this is the min longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v2/list-for-rent"
    querystring = {'limit': limit, 'city': city, 'state_code': state_code, 'offset': offset, }
    if price_max:
        querystring['price_max'] = price_max
    if baths_min:
        querystring['baths_min'] = baths_min
    if beds_min:
        querystring['beds_min'] = beds_min
    if allows_cats:
        querystring['allows_cats'] = allows_cats
    if lot_sqft_min:
        querystring['lot_sqft_min'] = lot_sqft_min
    if features:
        querystring['features'] = features
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    if lot_sqft_max:
        querystring['lot_sqft_max'] = lot_sqft_max
    if radius:
        querystring['radius'] = radius
    if lat_min:
        querystring['lat_min'] = lat_min
    if allows_dogs:
        querystring['allows_dogs'] = allows_dogs
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if price_min:
        querystring['price_min'] = price_min
    if postal_code:
        querystring['postal_code'] = postal_code
    if prop_sub_type:
        querystring['prop_sub_type'] = prop_sub_type
    if prop_type:
        querystring['prop_type'] = prop_type
    if sort:
        querystring['sort'] = sort
    if lng_max:
        querystring['lng_max'] = lng_max
    if lat_max:
        querystring['lat_max'] = lat_max
    if lng_min:
        querystring['lng_min'] = lng_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_auto_complete_deprecated(input: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by city, ward, street name to pass in other endpoints. This endpoint also helps to get a specific property id by its address"
    input: Name of cities, districts, places
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/locations/auto-complete"
    querystring = {'input': input, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_sold_deprecated(limit: int, city: str, offset: int, state_code: str, age_min: int=None, sqft_min: int=None, lng_max: int=None, prop_type: str=None, lot_sqft_max: int=None, sqft_max: int=None, radius: int=None, postal_code: str=None, price_min: int=None, sort: str='relevance', lat_max: int=None, lat_min: int=None, lot_sqft_min: int=None, price_max: int=None, age_max: int=None, lng_min: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List sold properties"
    limit: The number of items to be responded in every request
        city: The value of city field responded in locations/auto-complete API (do not use this parameter with postal_code)
        offset: The offset of items to be ignored in response for paging
        state_code: The value of state&#95;code field responded in locations/auto-complete API (do not use this parameter with postal&#95;code)
        age_min: Min age of home
        sqft_min: Min size of the properties
        lng_max: Look for properties in bounding box, this is the max longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        prop_type: One of the followings (separated by comma for multiple values): single&#95;family, condo, mobile, multi&#95;family, farm, land
        lot_sqft_max: Max Lot/Acreage size
        sqft_max: Max size of the properties
        radius: Radius in miles to look for properties
        postal_code: Zip code or postal code (do not use this parameter with city and state_code)
        price_min: Option filter by setting min price
        sort: One of the followings : price&#95;low|price&#95;high
        lat_max: Look for properties in bounding box, this is the max latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lat_min: Look for properties in bounding box, this is the min latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lot_sqft_min: Min Lot/Acreage size
        price_max: Option filter by setting max price
        age_max: Max age of home
        lng_min: Look for properties in bounding box, this is the min longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/list-sold"
    querystring = {'limit': limit, 'city': city, 'offset': offset, 'state_code': state_code, }
    if age_min:
        querystring['age_min'] = age_min
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if lng_max:
        querystring['lng_max'] = lng_max
    if prop_type:
        querystring['prop_type'] = prop_type
    if lot_sqft_max:
        querystring['lot_sqft_max'] = lot_sqft_max
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    if radius:
        querystring['radius'] = radius
    if postal_code:
        querystring['postal_code'] = postal_code
    if price_min:
        querystring['price_min'] = price_min
    if sort:
        querystring['sort'] = sort
    if lat_max:
        querystring['lat_max'] = lat_max
    if lat_min:
        querystring['lat_min'] = lat_min
    if lot_sqft_min:
        querystring['lot_sqft_min'] = lot_sqft_min
    if price_max:
        querystring['price_max'] = price_max
    if age_max:
        querystring['age_max'] = age_max
    if lng_min:
        querystring['lng_min'] = lng_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finance_rates(loc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get finance rates"
    loc: The postal code
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/finance/rates"
    querystring = {'loc': loc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_v2_auto_complete(input: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by city, ward, street name to pass in other endpoints. This endpoint also helps to get a specific property id by its address"
    input: States, cities, districts, addresses, zipcode. 
Ex : 
California
Los Angeles
2425 Sahalee Dr W Sammamish, WA
        limit: The number of items per response, for paging purpose
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/locations/v2/auto-complete"
    querystring = {'input': input, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mortgage_calculate_affordability(loan_term: int, homeowner_insurance_rate: int, down_payment: int, debt_to_income_ratio: int, tax_rate: int, interest_rate: int, monthly_debt: int, annual_income: int, is_pmi_included: bool=None, hoa_fees: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate affordability"
    is_pmi_included: Served in the U.S. Military. If the value is false, down_payment will be zero.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/mortgage/calculate-affordability"
    querystring = {'loan_term': loan_term, 'homeowner_insurance_rate': homeowner_insurance_rate, 'down_payment': down_payment, 'debt_to_income_ratio': debt_to_income_ratio, 'tax_rate': tax_rate, 'interest_rate': interest_rate, 'monthly_debt': monthly_debt, 'annual_income': annual_income, }
    if is_pmi_included:
        querystring['is_pmi_included'] = is_pmi_included
    if hoa_fees:
        querystring['hoa_fees'] = hoa_fees
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mortgage_v2_calculate(price: int, rate: int, down_payment: int, term: int, hoa_fees: int, home_insurance: int, property_tax_rate: int, apply_veterans_benefits: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate mortgage"
    price: Home price
        rate: Interest rate
        down_payment: Down payment
        term: Mortgage loan type
        hoa_fees: Home owner association fee
        home_insurance: Home insurance
        property_tax_rate: Property tax
        apply_veterans_benefits: Whether or not apply veterans benefits
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/mortgage/v2/calculate"
    querystring = {'price': price, 'rate': rate, 'down_payment': down_payment, 'term': term, 'hoa_fees': hoa_fees, 'home_insurance': home_insurance, 'property_tax_rate': property_tax_rate, }
    if apply_veterans_benefits:
        querystring['apply_veterans_benefits'] = apply_veterans_benefits
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mortgage_v2_check_rates(postal_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check mortgage rates"
    postal_code: The postal code
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/mortgage/v2/check-rates"
    querystring = {'postal_code': postal_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schools_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a school"
    id: The value of schools -> id field returned in .../schools/list endpoint
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/schools/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schools_list(state_code: str, education_level: str='elementary', offset: int=0, limit: int=10, city: str=None, postal_code: str=None, school_district_id: str=None, county: str=None, neighborhood: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List schools with options and filters"
    state_code: Filter schools by state (You need to specify at least one of the following : state&#95;code|city|county|neighborhood|postal&#95;code|school&#95;district&#95;id)
        education_level: One of the following : elementary|high|middle|private
        offset: The offset of records to ignore, for paging purpose
        limit: The number of items per response, for paging purpose
        city: Filter schools by city (You need to specify at least one of the following : state&#95;code|city|county|neighborhood|postal&#95;code|school&#95;district&#95;id)
        postal_code: Filter schools by postal_code (You need to specify at least one of the following : state&#95;code|city|county|neighborhood|postal&#95;code|school&#95;district&#95;id)
        school_district_id: Filter schools by school_district_id (You need to specify at least one of the following : state&#95;code|city|county|neighborhood|postal&#95;code|school&#95;district&#95;id)
        county: Filter schools by county (You need to specify at least one of the following : state&#95;code|city|county|neighborhood|postal&#95;code|school&#95;district&#95;id)
        neighborhood: Filter schools by neighborhood (You need to specify at least one of the following : state&#95;code|city|county|neighborhood|postal&#95;code|school&#95;district&#95;id)
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/schools/list"
    querystring = {'state_code': state_code, }
    if education_level:
        querystring['education_level'] = education_level
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if city:
        querystring['city'] = city
    if postal_code:
        querystring['postal_code'] = postal_code
    if school_district_id:
        querystring['school_district_id'] = school_district_id
    if county:
        querystring['county'] = county
    if neighborhood:
        querystring['neighborhood'] = neighborhood
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schools_get_school_district(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a school district"
    id: The value of districts -> id field returned in .../schools/list endpoint
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/schools/get-school-district"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_for_sale_deprecated(limit: int, offset: int, city: str, state_code: str, beds_min: int=None, is_pending: bool=None, lot_sqft_max: int=None, sort: str='relevance', age_min: int=None, lng_max: int=None, lat_max: int=None, price_min: int=None, sqft_max: int=None, baths_min: int=None, lot_sqft_min: int=None, is_new_construction: bool=None, listed_date_min: str=None, postal_code: str=None, prop_type: str=None, price_max: int=None, sqft_min: int=None, radius: int=None, reduced: str=None, age_max: int=None, lng_min: int=None, lat_min: int=None, is_foreclosure: bool=None, is_contingent: bool=None, features: str=None, is_matterports: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties for sale"
    limit: The number of items to be responded in every request
        offset: The offset of items to be ignored in response for paging
        city: The value of city field responded in locations/auto-complete API (do not use this parameter with postal_code)
        state_code: The value of state&#95;code field responded in locations/auto-complete API (do not use this parameter with postal&#95;code)
        beds_min: Option filter by setting at least the number of bedrooms
        is_pending: true/false - Pending only
        lot_sqft_max: Max Lot/Acreage size
        sort: One of the followings (relevance | price&#95;low | price&#95;high | photos | newest | open&#95;house&#95;date | sqft&#95;high | price&#95;reduced&#95;date)
        age_min: Min age of home
        lng_max: Look for properties in bounding box, this is the max longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lat_max: Look for properties in bounding box, this is the max latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        price_min: Option filter by setting min price
        sqft_max: Max size of the properties
        baths_min: Option filter by setting at least the number of bathrooms
        lot_sqft_min: Min Lot/Acreage size
        is_new_construction: true/false - New construction only
        listed_date_min: Date string format yyyy-MM-dd'T'HH:mm:ss'Z' , such as : 2019-08-01T16:24:40Z .The date from which the properties have been on realtor.com
        postal_code: Zip code or postal code (do not use this parameter with city and state_code)
        prop_type: One of the followings (separated by comma for multiple values): single&#95;family,condo,mobile,multi&#95;family,farm,land
        price_max: Option filter by setting max price
        sqft_min: Min size of the properties
        radius: Radius in miles to look for properties
        reduced: true/false - Price reduced only
        age_max: Max age of home
        lng_min: Look for properties in bounding box, this is the min longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lat_min: Look for properties in bounding box, this is the min latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        is_foreclosure: true/false - Foreclosures only
        is_contingent: true/false - Contingent only
        features: One of the followings (separated by comma for multiple values): basement,den&#95;or&#95;office,dining&#95;room,family&#95;room,game&#95;room,washer&#95;dryer,energy&#95;efficient&#95;home,central&#95;air,central&#95;heat,forced&#95;air,carport,garage&#95;1&#95;or&#95;more,garage&#95;2&#95;or&#95;more,garage&#95;3&#95;or&#95;more,rv&#95;or&#95;boat&#95;parking,disability&#95;features,fireplace,hardwood&#95;floors,horse&#95;facilities,spa&#95;or&#95;hot&#95;tub,swimming&#95;pool,tennis&#95;court,single&#95;story,two&#95;or&#95;more&#95;stories,lease&#95;option,pets&#95;allowed,corner&#95;lot,cul&#95;de&#95;sac,golf&#95;course&#95;lot&#95;or&#95;frontage,waterfront,city&#95;view,golf&#95;course&#95;view,hill&#95;mountain,lake&#95;view,ocean&#95;view,river&#95;view,water&#95;view,view,community&#95;swimming&#95;pool,community&#95;boat&#95;facilities,community&#95;spa&#95;or&#95;hot&#95;tub,community&#95;tennis&#95;court,community&#95;golf,community&#95;clubhouse,community&#95;security&#95;features,senior&#95;community,community&#95;horse&#95;facilities,community&#95;park,recreation&#95;facilities,exercise&#95;area
        is_matterports: true/false - 3D tours only
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/list-for-sale"
    querystring = {'limit': limit, 'offset': offset, 'city': city, 'state_code': state_code, }
    if beds_min:
        querystring['beds_min'] = beds_min
    if is_pending:
        querystring['is_pending'] = is_pending
    if lot_sqft_max:
        querystring['lot_sqft_max'] = lot_sqft_max
    if sort:
        querystring['sort'] = sort
    if age_min:
        querystring['age_min'] = age_min
    if lng_max:
        querystring['lng_max'] = lng_max
    if lat_max:
        querystring['lat_max'] = lat_max
    if price_min:
        querystring['price_min'] = price_min
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    if baths_min:
        querystring['baths_min'] = baths_min
    if lot_sqft_min:
        querystring['lot_sqft_min'] = lot_sqft_min
    if is_new_construction:
        querystring['is_new_construction'] = is_new_construction
    if listed_date_min:
        querystring['listed_date_min'] = listed_date_min
    if postal_code:
        querystring['postal_code'] = postal_code
    if prop_type:
        querystring['prop_type'] = prop_type
    if price_max:
        querystring['price_max'] = price_max
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if radius:
        querystring['radius'] = radius
    if reduced:
        querystring['reduced'] = reduced
    if age_max:
        querystring['age_max'] = age_max
    if lng_min:
        querystring['lng_min'] = lng_min
    if lat_min:
        querystring['lat_min'] = lat_min
    if is_foreclosure:
        querystring['is_foreclosure'] = is_foreclosure
    if is_contingent:
        querystring['is_contingent'] = is_contingent
    if features:
        querystring['features'] = features
    if is_matterports:
        querystring['is_matterports'] = is_matterports
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v3_detail(property_id: str, listing_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detail information (include new Home Value feature)"
    property_id: The value of property&#95;id field returned in .../properties/.../list endpoint
        listing_id: The value of listing&#95;id field returned in .../properties/.../list endpoint
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v3/detail"
    querystring = {'property_id': property_id, }
    if listing_id:
        querystring['listing_id'] = listing_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v3_get_surroundings(property_id: str, enable_flood: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get surroundings data around  a property"
    property_id: The value of property&#95;id field returned in .../properties/.../list endpoint
        enable_flood: Whether or not include flood information
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v3/get-surroundings"
    querystring = {'property_id': property_id, }
    if enable_flood:
        querystring['enable_flood'] = enable_flood
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v3_list_similar_homes(property_id: str, limit: int=10, status: str='for_sale', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar homes"
    property_id: The value of property_id field returned in .../properties/.../list endpoint
        limit: The number of items per response, for paging purpose
        status: One of the following : for&#95;sale|ready&#95;to&#95;build|for&#95;rent|sold|off&#95;market|other|new&#95;community
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v3/list-similar-homes"
    querystring = {'property_id': property_id, }
    if limit:
        querystring['limit'] = limit
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_list_similar_homes_deprecated(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar properties for sale
		* This endpoint is deprecating, the official have changed to use .../properties/v2/list.... endpoints to list similar properties."
    property_id: The value of property&#95;id field returned in  .../properties/list-.... or  .../properties/v2/list-... endpoints.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v2/list-similar-homes"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v3_get_photos(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get photos of a property"
    property_id: The value of property&#95;id field returned in .../properties/.../list endpoint
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v3/get-photos"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_detail_deprecated(property_id: str, listing_id: str, prop_status: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detail information"
    property_id: The value of property_id field returned in  .../properties/list-.... or  .../properties/v2/list-... endpoints.
        listing_id: The value of listing_id field returned in  .../properties/list-.... or  .../properties/v2/list-... endpoints.
        prop_status: One of the followings : for&#95;sale|for&#95;rent
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/detail"
    querystring = {'property_id': property_id, 'listing_id': listing_id, 'prop_status': prop_status, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v3_get_commute_time(destination_address: str, property_id: str, transportation_type: str='walking', with_traffic: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get commute time to travel to a location"
    destination_address: An address, you should use .../locations/v2/auto-complete to get a complete and correct address . Ex : 7830 Poppy Blvd
        property_id: The value of property&#95;id field returned in .../properties/.../list endpoint
        transportation_type: One of the following : bicycling|driving|transit|walking
        with_traffic: Whether or not calculate the time in rush hours
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v3/get-commute-time"
    querystring = {'destination_address': destination_address, 'property_id': property_id, }
    if transportation_type:
        querystring['transportation_type'] = transportation_type
    if with_traffic:
        querystring['with_traffic'] = with_traffic
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_list_sold_deprecated(offset: int, limit: int, city: str, state_code: str, postal_code: str=None, sort: str='sold_date', price_min: int=None, beds_min: int=None, prop_type: str=None, radius: int=None, age_max: int=None, price_max: int=None, lot_sqft_max: int=None, baths_min: int=None, age_min: int=None, lat_max: int=None, sqft_min: int=None, lot_sqft_min: int=None, sqft_max: int=None, lng_max: int=None, lat_min: int=None, lng_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List sold properties"
    offset: The offset of items to be ignored in response for paging
        limit: The number of items to be responded in every request
        city: The value of city field responded in locations/auto-complete API (do not use this parameter with postal_code)
        state_code: The value of state&#95;code field responded in locations/auto-complete API (do not use this parameter with postal&#95;code)
        postal_code: Zip code or postal code (do not use this parameter with city and state_code)
        sort: One of the followings : sold&#95;date|beds&#95;high|price&#95;low|price&#95;high|lot&#95;sqft&#95;high
        price_min: Option filter by setting min price
        beds_min: Min beds of properties
        prop_type: One of the followings (separated by comma for multiple values): single&#95;family,multi&#95;family,condo,mobile,land,farm,other
        radius: Radius in miles to look for properties (1 to 20)
        age_max: Max age of home
        price_max: Option filter by setting max price
        lot_sqft_max: Max Lot/Acreage size
        baths_min: Min baths of properties
        age_min: Min age of home
        lat_max: Look for properties in bounding box, this is the max latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        sqft_min: Min size of the properties
        lot_sqft_min: Min Lot/Acreage size
        sqft_max: Max size of the properties
        lng_max: Look for properties in bounding box, this is the max longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lat_min: Look for properties in bounding box, this is the min latitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        lng_min: Look for properties in bounding box, this is the min longitude of the coordinate.
Has no affect if postal&#95;code, or city or state&#95;code parameter has value.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v2/list-sold"
    querystring = {'offset': offset, 'limit': limit, 'city': city, 'state_code': state_code, }
    if postal_code:
        querystring['postal_code'] = postal_code
    if sort:
        querystring['sort'] = sort
    if price_min:
        querystring['price_min'] = price_min
    if beds_min:
        querystring['beds_min'] = beds_min
    if prop_type:
        querystring['prop_type'] = prop_type
    if radius:
        querystring['radius'] = radius
    if age_max:
        querystring['age_max'] = age_max
    if price_max:
        querystring['price_max'] = price_max
    if lot_sqft_max:
        querystring['lot_sqft_max'] = lot_sqft_max
    if baths_min:
        querystring['baths_min'] = baths_min
    if age_min:
        querystring['age_min'] = age_min
    if lat_max:
        querystring['lat_max'] = lat_max
    if sqft_min:
        querystring['sqft_min'] = sqft_min
    if lot_sqft_min:
        querystring['lot_sqft_min'] = lot_sqft_min
    if sqft_max:
        querystring['sqft_max'] = sqft_max
    if lng_max:
        querystring['lng_max'] = lng_max
    if lat_min:
        querystring['lat_min'] = lat_min
    if lng_min:
        querystring['lng_min'] = lng_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mortgage_check_equity_rates(creditscore: str, zip: str, loanamount: int, propertyvalue: str, loanproduct: str, mortgagebalance: int, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check mortgage equity rates"
    creditscore: One of the following : excellent|good|fair|poor
        zip: The postal code
        loanproduct: One of the following : HELOC,HELOAN&#95;FIXED&#95;5YEARS,HELOAN&#95;FIXED&#95;10YEARS,HELOAN&#95;FIXED&#95;15YEARS,HELOAN&#95;FIXED&#95;20YEARS,HELOAN&#95;FIXED&#95;30YEARS. Separate by comma for multiple applied values
        state: The state code
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/mortgage/check-equity-rates"
    querystring = {'creditScore': creditscore, 'zip': zip, 'loanAmount': loanamount, 'propertyValue': propertyvalue, 'loanProduct': loanproduct, 'mortgageBalance': mortgagebalance, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_get_listings(fulfillment_id: int, is_id: str, agent_id: int, page: int=1, type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's listings"
    fulfillment_id: The value of advertiser_id field returned in .../agents/list endpoint
        id: The value of abbreviation field returned in .../agents/list endpoint
        agent_id: The value of ...member/id JSON object returned in .../agents/list endpoint
        page: For paging purpose
        type: One of the following : all|forSale|forSold|forRent|openHouses
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/agents/get-listings"
    querystring = {'fulfillment_id': fulfillment_id, 'id': is_id, 'agent_id': agent_id, }
    if page:
        querystring['page'] = page
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_get_reviews(advertiser_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent reviews"
    advertiser_id: The value of advertiser_id field returned in .../agents/list endpoint
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/agents/get-reviews"
    querystring = {'advertiser_id': advertiser_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_get_profile(nrds_id: str, advertiser_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent profile"
    nrds_id: The value of nrds_id field returned in .../agents/list endpoint
        advertiser_id: The value of advertiser_id field returned in .../agents/list endpoint
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/agents/get-profile"
    querystring = {'nrds_id': nrds_id, 'advertiser_id': advertiser_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schools_list_nearby_deprecated(lon: int, lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List schools that are near the property"
    lon: The longitude co-ordinate
        lat: The latitude co-ordinate
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/schools/list-nearby"
    querystring = {'lon': lon, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_list(postal_code: str, recommendations_count_min: int=None, sort: str='recent_activity_high', agent_rating_min: int=None, offset: int=0, limit: int=20, name: str=None, agent_type: str=None, types: str='agent', price_min: int=None, price_max: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List agents with filters and options"
    postal_code: Zip code or postal code
        recommendations_count_min: Number of recommendations (max 10)
        sort: One of the followings : recent&#95;activity&#95;high|recently&#95;sold&#95;high|for&#95;sale&#95;count&#95;high|recommendations&#95;count&#95;high|agent&#95;rating&#95;high
        agent_rating_min: Rating (max 5)
        offset: The offset of items to be ignored in response for paging
        limit: For paging purpose (max 20)
        name: Search for agent or team or company by name
        agent_type: One of the followings or leave empty : buyer | seller
        types: One of the followings : agent | team | office
        price_min: Option filter by setting min price
        price_max: Option filter by setting max price
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/agents/list"
    querystring = {'postal_code': postal_code, }
    if recommendations_count_min:
        querystring['recommendations_count_min'] = recommendations_count_min
    if sort:
        querystring['sort'] = sort
    if agent_rating_min:
        querystring['agent_rating_min'] = agent_rating_min
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    if agent_type:
        querystring['agent_type'] = agent_type
    if types:
        querystring['types'] = types
    if price_min:
        querystring['price_min'] = price_min
    if price_max:
        querystring['price_max'] = price_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_list_similar_rental_homes_deprecated(postal_code: str, property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar properties for rent
		* This endpoint is deprecating, the official have changed to use .../properties/v2/list.... endpoints to list similar properties."
    postal_code: The value of postal&#95;code field returned in all list endpoints
        property_id: The value of property&#95;id field returned in  .../properties/list-.... or  .../properties/v2/list-... endpoints.
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/v2/list-similar-rental-homes"
    querystring = {'postal_code': postal_code, 'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_similarities_deprecated(prop_status: str, property_id: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar properties"
    prop_status: One of the followings : for&#95;sale|for&#95;rent
        property_id: The value of property&#95;id field returned in  .../properties/list-.... or  .../properties/v2/list-... endpoints.
        limit: The number of items responded 
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/list-similarities"
    querystring = {'prop_status': prop_status, 'property_id': property_id, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_by_mls_deprecated(mls_id: str, offset: int=0, sort: str='price_low', limit: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties by MLS ID"
    mls_id: The MLS ID
        offset: The offset of items to be ignored in response for paging
        sort: One of the followings : price&#95;low|price&#95;high|photos|newest|open&#95;house&#95;date
        limit: The number of items per response
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/properties/list-by-mls"
    querystring = {'mls_id': mls_id, }
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_get_recommendations(advertiser_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent's recommendations"
    advertiser_id: The value of advertiser_id field returned in .../agents/list endpoint
        
    """
    url = f"https://realty-in-us.p.rapidapi.com/agents/get-recommendations"
    querystring = {'advertiser_id': advertiser_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

