import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def property_details(property_id: str='1497548641', address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property details by  property ID  or by Address"
    
    """
    url = f"https://realtor16.p.rapidapi.com/property"
    querystring = {}
    if property_id:
        querystring['property_id'] = property_id
    if address:
        querystring['address'] = address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_region(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location suggestions by keyword"
    
    """
    url = f"https://realtor16.p.rapidapi.com/suggestion"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_agents(location: str, page: int=1, price: str=None, agentname: str=None, lang: str=None, photo: bool=None, rating: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search agents by city/zip"
    price: Price range
example for range between 50k to 2m : 
50000_2000000
        lang: [afrikaans
albanian
arabic
armenian
asl
bengali
bosnian
bulgarian
chinese
croatian
czech
Danish
dutch
Estonian
farsi
filipino
finnish
french
gaelic
georgian
german
greek
hawaiian
hebrew
hindi
hungarian
indonesian
italian
japanese
korean
lao
latvian
lithuanian
malay
mandarin
nepali
norwegian
polish
portuguese
punjabi
romanian
russian
serbian
sindhi
singhalese
slovenian
spanish
swahili
swedish
tagalog
taiwanese
thai
turkish
ukrainian
urdu
vietnamese
yugoslavian]
        photo: If the agent has a photo or not 
1 if yes
0 if no
        rating: Rating (between 1 and 5)
        
    """
    url = f"https://realtor16.p.rapidapi.com/search_agents"
    querystring = {'location': location, }
    if page:
        querystring['page'] = page
    if price:
        querystring['price'] = price
    if agentname:
        querystring['agentname'] = agentname
    if lang:
        querystring['lang'] = lang
    if photo:
        querystring['photo'] = photo
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get agent details by id"
    
    """
    url = f"https://realtor16.p.rapidapi.com/agent"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_properties_for_sale(location: str, beds_min: int=None, baths_max: int=None, year_built_max: int=None, year_built_min: int=None, list_date_min: str=None, open_house_max: str=None, has_tour: bool=None, list_price_min: int=None, hoa_fee_optional_max: int=None, list_date_max: str=None, list_price_max: int=None, baths_min: int=None, open_house_min: str=None, type: str='single_family,condos', sort: str=None, beds_max: int=None, page: int=None, lot_sqft_min: int=None, lot_sqft_max: int=None, hoa_fee_optional_min: int=None, sqft_max: int=None, sqft_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search properties for sale by Address, School, City, Zip or Neighborhood. Filter the results using different parameter filters"
    type: Property type comma-separated or empty for all types:
single_family
multi_family
land
townhomes
duplex_triplex
mobile
condos
condo_townhome_rowhome_coop
condo_townhome
farm

        sort: Sort properties by :
Newest_Listing 
Highest_Price 
Lowest_Price
Open_House_Date
Recently-Reduced
Largest_Sqft
Lot_Size
        
    """
    url = f"https://realtor16.p.rapidapi.com/forsale"
    querystring = {'location': location, }
    if beds_min:
        querystring['beds-min'] = beds_min
    if baths_max:
        querystring['baths-max'] = baths_max
    if year_built_max:
        querystring['year_built-max'] = year_built_max
    if year_built_min:
        querystring['year_built-min'] = year_built_min
    if list_date_min:
        querystring['list_date-min'] = list_date_min
    if open_house_max:
        querystring['open_house-max'] = open_house_max
    if has_tour:
        querystring['has_tour'] = has_tour
    if list_price_min:
        querystring['list_price-min'] = list_price_min
    if hoa_fee_optional_max:
        querystring['hoa_fee_optional-max'] = hoa_fee_optional_max
    if list_date_max:
        querystring['list_date-max'] = list_date_max
    if list_price_max:
        querystring['list_price-max'] = list_price_max
    if baths_min:
        querystring['baths-min'] = baths_min
    if open_house_min:
        querystring['open_house-min'] = open_house_min
    if type:
        querystring['type'] = type
    if sort:
        querystring['sort'] = sort
    if beds_max:
        querystring['beds-max'] = beds_max
    if page:
        querystring['page'] = page
    if lot_sqft_min:
        querystring['lot_sqft-min'] = lot_sqft_min
    if lot_sqft_max:
        querystring['lot_sqft-max'] = lot_sqft_max
    if hoa_fee_optional_min:
        querystring['hoa_fee_optional-min'] = hoa_fee_optional_min
    if sqft_max:
        querystring['sqft-max'] = sqft_max
    if sqft_min:
        querystring['sqft-min'] = sqft_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_properties_for_rent(location: str, move_in_date_max: str=None, sqft_min: int=None, type: str=None, sqft_max: int=None, list_price_min: int=None, page: int=None, beds_min: int=None, sort: str=None, threedtours: bool=None, baths_max: int=None, keyword: str=None, list_price_max: int=None, beds_max: int=None, baths_min: int=None, nofees: bool=None, dogs: bool=None, move_in_date_min: str=None, cats: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search properties for rent by location"
    type: Property type comma-separated or empty for all types:
single_family
apartment
townhomes
condo_townhome_rowhome_coop
duplex_triplex
condos
condo_townhome_rowhome_coop
condo_townhome

        sort: Sort properties by :
Recently_Added
Highest_Price 
Lowest_Price
        
    """
    url = f"https://realtor16.p.rapidapi.com/forrent"
    querystring = {'location': location, }
    if move_in_date_max:
        querystring['move_in_date-max'] = move_in_date_max
    if sqft_min:
        querystring['sqft-min'] = sqft_min
    if type:
        querystring['type'] = type
    if sqft_max:
        querystring['sqft-max'] = sqft_max
    if list_price_min:
        querystring['list_price-min'] = list_price_min
    if page:
        querystring['page'] = page
    if beds_min:
        querystring['beds-min'] = beds_min
    if sort:
        querystring['sort'] = sort
    if threedtours:
        querystring['threeDTours'] = threedtours
    if baths_max:
        querystring['baths-max'] = baths_max
    if keyword:
        querystring['keyword'] = keyword
    if list_price_max:
        querystring['list_price-max'] = list_price_max
    if beds_max:
        querystring['beds-max'] = beds_max
    if baths_min:
        querystring['baths-min'] = baths_min
    if nofees:
        querystring['noFees'] = nofees
    if dogs:
        querystring['dogs'] = dogs
    if move_in_date_min:
        querystring['move_in_date-min'] = move_in_date_min
    if cats:
        querystring['cats'] = cats
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtor16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

