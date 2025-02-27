import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def travel_time_search_to_rent(search_area: str, keywords: str=None, furnishing_state: str=None, show_let_or_let_agreed: bool=None, available_from: str=None, added_to_site: str=None, price_per: str=None, min_bedroom: int=None, has_garden: bool=None, has_bills_included: bool=None, has_parking_garage: bool=None, has_balcony_terrace: bool=None, has_pets_allowed: bool=None, max_travel_time: str=None, max_price: int=None, retirement_homes: str=None, sort_order: str=None, house_share: str=None, max_bedroom: int=None, transport_type: str=None, min_price: int=None, property_type: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Travel time search to rent"
    keywords: e.g. conservatory or \"double garage\"
        
    """
    url = f"https://zoopla3.p.rapidapi.com/to-rent/travel-time-search-to-rent"
    querystring = {'search_area': search_area, }
    if keywords:
        querystring['keywords'] = keywords
    if furnishing_state:
        querystring['furnishing_state'] = furnishing_state
    if show_let_or_let_agreed:
        querystring['show_let_or_let_agreed'] = show_let_or_let_agreed
    if available_from:
        querystring['available_from'] = available_from
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if price_per:
        querystring['price_per'] = price_per
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if has_garden:
        querystring['has_garden'] = has_garden
    if has_bills_included:
        querystring['has_bills_included'] = has_bills_included
    if has_parking_garage:
        querystring['has_parking_garage'] = has_parking_garage
    if has_balcony_terrace:
        querystring['has_balcony_terrace'] = has_balcony_terrace
    if has_pets_allowed:
        querystring['has_pets_allowed'] = has_pets_allowed
    if max_travel_time:
        querystring['max_travel_time'] = max_travel_time
    if max_price:
        querystring['max_price'] = max_price
    if retirement_homes:
        querystring['retirement_homes'] = retirement_homes
    if sort_order:
        querystring['sort_order'] = sort_order
    if house_share:
        querystring['house_share'] = house_share
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if transport_type:
        querystring['transport_type'] = transport_type
    if min_price:
        querystring['min_price'] = min_price
    if property_type:
        querystring['property_type'] = property_type
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def travel_time_search_for_sale(search_area: str, has_chain_free: bool=None, added_to_site: str=None, has_reduced_price: bool=None, keywords: str=None, has_under_offer_sold_stc: bool=None, has_parking_garage: bool=None, has_garden: bool=None, has_balcony_terrace: bool=None, shared_ownership: str=None, auction: str=None, retirement_homes: str=None, new_homes: str=None, property_type: str=None, max_price: int=None, max_bedroom: int=None, transport_type: str=None, min_bedroom: int=None, min_price: int=None, max_travel_time: str=None, sort_order: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Travel time search for sale"
    keywords: e.g. conservatory or \\\"double garage\\\"
        
    """
    url = f"https://zoopla3.p.rapidapi.com/for-sale/travel-time-search-for-sale"
    querystring = {'search_area': search_area, }
    if has_chain_free:
        querystring['has_chain_free'] = has_chain_free
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if has_reduced_price:
        querystring['has_reduced_price'] = has_reduced_price
    if keywords:
        querystring['keywords'] = keywords
    if has_under_offer_sold_stc:
        querystring['has_under_offer_sold_stc'] = has_under_offer_sold_stc
    if has_parking_garage:
        querystring['has_parking_garage'] = has_parking_garage
    if has_garden:
        querystring['has_garden'] = has_garden
    if has_balcony_terrace:
        querystring['has_balcony_terrace'] = has_balcony_terrace
    if shared_ownership:
        querystring['shared_ownership'] = shared_ownership
    if auction:
        querystring['auction'] = auction
    if retirement_homes:
        querystring['retirement_homes'] = retirement_homes
    if new_homes:
        querystring['new_homes'] = new_homes
    if property_type:
        querystring['property_type'] = property_type
    if max_price:
        querystring['max_price'] = max_price
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if transport_type:
        querystring['transport_type'] = transport_type
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if min_price:
        querystring['min_price'] = min_price
    if max_travel_time:
        querystring['max_travel_time'] = max_travel_time
    if sort_order:
        querystring['sort_order'] = sort_order
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uk_property_to_rent(search_area: str, keywords: str=None, has_chain_free: bool=None, has_balcony_terrace: bool=None, shared_ownership: str=None, property_type: str=None, sort_order: str=None, max_price: str=None, max_bedroom: int=None, page: int=None, search_radius: str=None, has_reduced_price: bool=None, has_under_offer_sold_stc: bool=None, added_to_site: str=None, retirement_homes: str=None, has_parking_garage: bool=None, has_garden: bool=None, auction: str=None, min_price: int=None, new_homes: str=None, min_bedroom: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UK property to rent"
    keywords: e.g. conservatory or \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"double garage\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        property_type: Property type comma-separated or empty for all types
Ex: SemiDetached,Flats


`SemiDetached`
`Flats`
`FarmsLand`
`Detached`
`Terraced`
`Bungalows`
`ParkHomes`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/to-rent/property"
    querystring = {'search_area': search_area, }
    if keywords:
        querystring['keywords'] = keywords
    if has_chain_free:
        querystring['has_chain_free'] = has_chain_free
    if has_balcony_terrace:
        querystring['has_balcony_terrace'] = has_balcony_terrace
    if shared_ownership:
        querystring['shared_ownership'] = shared_ownership
    if property_type:
        querystring['property_type'] = property_type
    if sort_order:
        querystring['sort_order'] = sort_order
    if max_price:
        querystring['max_price'] = max_price
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if page:
        querystring['page'] = page
    if search_radius:
        querystring['search_radius'] = search_radius
    if has_reduced_price:
        querystring['has_reduced_price'] = has_reduced_price
    if has_under_offer_sold_stc:
        querystring['has_under_offer_sold_stc'] = has_under_offer_sold_stc
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if retirement_homes:
        querystring['retirement_homes'] = retirement_homes
    if has_parking_garage:
        querystring['has_parking_garage'] = has_parking_garage
    if has_garden:
        querystring['has_garden'] = has_garden
    if auction:
        querystring['auction'] = auction
    if min_price:
        querystring['min_price'] = min_price
    if new_homes:
        querystring['new_homes'] = new_homes
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_agents_commercial_agents(location: str, search_radius: str=None, type_of_agent: str=None, page: int=None, agent_name: str=None, sort_order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "find-agents/commercial-agents"
    type_of_agent: Default: `CommercialAgents`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/find-agents/commercial-agents"
    querystring = {'location': location, }
    if search_radius:
        querystring['search_radius'] = search_radius
    if type_of_agent:
        querystring['type_of_agent'] = type_of_agent
    if page:
        querystring['page'] = page
    if agent_name:
        querystring['agent_name'] = agent_name
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_agents_letting_agents(location: str, sort_order: str=None, search_radius: str=None, page: int=None, agent_name: str=None, type_of_agent: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "find-agents/letting-agents"
    type_of_agent: Default: `LettingAgents`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/find-agents/letting-agents"
    querystring = {'location': location, }
    if sort_order:
        querystring['sort_order'] = sort_order
    if search_radius:
        querystring['search_radius'] = search_radius
    if page:
        querystring['page'] = page
    if agent_name:
        querystring['agent_name'] = agent_name
    if type_of_agent:
        querystring['type_of_agent'] = type_of_agent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_agents_estate_agents(location: str, sort_order: str=None, type_of_agent: str=None, page: int=None, search_radius: str=None, agent_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "find-agents/estate-agents"
    type_of_agent: Default: `EstateAgents`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/find-agents/estate-agents"
    querystring = {'location': location, }
    if sort_order:
        querystring['sort_order'] = sort_order
    if type_of_agent:
        querystring['type_of_agent'] = type_of_agent
    if page:
        querystring['page'] = page
    if search_radius:
        querystring['search_radius'] = search_radius
    if agent_name:
        querystring['agent_name'] = agent_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_agents_auto_complete(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "find-agents/auto-complete"
    
    """
    url = f"https://zoopla3.p.rapidapi.com/find-agents/auto-complete"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_build_developments(search_area: str, max_bedroom: int=None, min_price: int=None, page: int=None, min_bedroom: int=None, max_price: int=None, property_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New build developments"
    property_type: Property type comma-separated or empty for all types:
Ex: SemiDetached,Flats

`SemiDetached`
`Flats`
`FarmsLand`
`Detached`
`Terraced`
`Bungalows`
`ParkHomes`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/new-homes/new-build-developments"
    querystring = {'search_area': search_area, }
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if min_price:
        querystring['min_price'] = min_price
    if page:
        querystring['page'] = page
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if max_price:
        querystring['max_price'] = max_price
    if property_type:
        querystring['property_type'] = property_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_homes_for_sale(search_area: str, has_under_offer_sold_stc: bool=None, has_parking_garage: bool=None, retirement_homes: str=None, auction: str=None, max_price: str=None, property_type: str=None, min_bedroom: int=None, sort_order: str=None, shared_ownership: str=None, max_bedroom: int=None, min_price: int=None, search_radius: str=None, has_balcony_terrace: bool=None, page: int=None, has_reduced_price: bool=None, added_to_site: str=None, new_homes: str=None, has_garden: bool=None, has_chain_free: bool=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New homes for sale"
    property_type: Property type comma-separated or empty for all types
Ex: SemiDetached,Flats


`SemiDetached`
`Flats`
`FarmsLand`
`Detached`
`Terraced`
`Bungalows`
`ParkHomes`
        new_homes: Default: `ShowOnly`
        keywords: e.g. conservatory or \\\"double garage\\\"
        
    """
    url = f"https://zoopla3.p.rapidapi.com/new-homes/new-homes-for-sale"
    querystring = {'search_area': search_area, }
    if has_under_offer_sold_stc:
        querystring['has_under_offer_sold_stc'] = has_under_offer_sold_stc
    if has_parking_garage:
        querystring['has_parking_garage'] = has_parking_garage
    if retirement_homes:
        querystring['retirement_homes'] = retirement_homes
    if auction:
        querystring['auction'] = auction
    if max_price:
        querystring['max_price'] = max_price
    if property_type:
        querystring['property_type'] = property_type
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if sort_order:
        querystring['sort_order'] = sort_order
    if shared_ownership:
        querystring['shared_ownership'] = shared_ownership
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if min_price:
        querystring['min_price'] = min_price
    if search_radius:
        querystring['search_radius'] = search_radius
    if has_balcony_terrace:
        querystring['has_balcony_terrace'] = has_balcony_terrace
    if page:
        querystring['page'] = page
    if has_reduced_price:
        querystring['has_reduced_price'] = has_reduced_price
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if new_homes:
        querystring['new_homes'] = new_homes
    if has_garden:
        querystring['has_garden'] = has_garden
    if has_chain_free:
        querystring['has_chain_free'] = has_chain_free
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_for_detail(uprn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "House-prices for detail"
    uprn: uprn from [House prices & values] endpoint
        
    """
    url = f"https://zoopla3.p.rapidapi.com/house-prices/detail"
    querystring = {'uprn': uprn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_values(search_area: str, sort_order: str=None, last_sold_within: str=None, page: int=None, property_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "House prices & values"
    last_sold_within: Unit: Month

60=5 years
120=10 years
240=20 years
        
    """
    url = f"https://zoopla3.p.rapidapi.com/house-prices/house-prices-values"
    querystring = {'search_area': search_area, }
    if sort_order:
        querystring['sort_order'] = sort_order
    if last_sold_within:
        querystring['last_sold_within'] = last_sold_within
    if page:
        querystring['page'] = page
    if property_type:
        querystring['property_type'] = property_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_detail(listing_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "property/detail"
    listing_id: `listing_id`: listingId from [UK property for sale],[UK new homes for sale],[UK property to rent] endpoints
        
    """
    url = f"https://zoopla3.p.rapidapi.com/property/detail"
    querystring = {'listing_id': listing_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uk_letting_agents(location: str, type_of_agent: str=None, sort_order: str=None, page: int=None, agent_name: str=None, search_radius: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UK letting agents"
    type_of_agent: Default: `LettingAgents`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/to-rent/letting-agents"
    querystring = {'location': location, }
    if type_of_agent:
        querystring['type_of_agent'] = type_of_agent
    if sort_order:
        querystring['sort_order'] = sort_order
    if page:
        querystring['page'] = page
    if agent_name:
        querystring['agent_name'] = agent_name
    if search_radius:
        querystring['search_radius'] = search_radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uk_estate_agents(location: str, search_radius: str=None, page: int=None, agent_name: str=None, sort_order: str=None, type_of_agent: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UK estate agents"
    type_of_agent: Default: `EstateAgents`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/for-sale/estate-agents"
    querystring = {'location': location, }
    if search_radius:
        querystring['search_radius'] = search_radius
    if page:
        querystring['page'] = page
    if agent_name:
        querystring['agent_name'] = agent_name
    if sort_order:
        querystring['sort_order'] = sort_order
    if type_of_agent:
        querystring['type_of_agent'] = type_of_agent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uk_new_homes_for_sale(search_area: str, has_under_offer_sold_stc: bool=None, has_reduced_price: bool=None, has_garden: bool=None, has_parking_garage: bool=None, has_balcony_terrace: bool=None, retirement_homes: str=None, has_chain_free: bool=None, auction: str=None, property_type: str=None, new_homes: str=None, page: int=None, sort_order: str=None, min_bedroom: int=None, search_radius: str=None, shared_ownership: str=None, added_to_site: str=None, min_price: int=None, max_bedroom: int=None, max_price: str=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UK new homes for sale"
    property_type: Property type comma-separated or empty for all types
Ex: SemiDetached,Flats


`SemiDetached`
`Flats`
`FarmsLand`
`Detached`
`Terraced`
`Bungalows`
`ParkHomes`
        new_homes: Default: `ShowOnly`
        keywords: e.g. conservatory or \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"double garage\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        
    """
    url = f"https://zoopla3.p.rapidapi.com/for-sale/new-homes"
    querystring = {'search_area': search_area, }
    if has_under_offer_sold_stc:
        querystring['has_under_offer_sold_stc'] = has_under_offer_sold_stc
    if has_reduced_price:
        querystring['has_reduced_price'] = has_reduced_price
    if has_garden:
        querystring['has_garden'] = has_garden
    if has_parking_garage:
        querystring['has_parking_garage'] = has_parking_garage
    if has_balcony_terrace:
        querystring['has_balcony_terrace'] = has_balcony_terrace
    if retirement_homes:
        querystring['retirement_homes'] = retirement_homes
    if has_chain_free:
        querystring['has_chain_free'] = has_chain_free
    if auction:
        querystring['auction'] = auction
    if property_type:
        querystring['property_type'] = property_type
    if new_homes:
        querystring['new_homes'] = new_homes
    if page:
        querystring['page'] = page
    if sort_order:
        querystring['sort_order'] = sort_order
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if search_radius:
        querystring['search_radius'] = search_radius
    if shared_ownership:
        querystring['shared_ownership'] = shared_ownership
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if min_price:
        querystring['min_price'] = min_price
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if max_price:
        querystring['max_price'] = max_price
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uk_property_for_sale(search_area: str, added_to_site: str=None, keywords: str=None, has_reduced_price: bool=None, max_price: str=None, has_balcony_terrace: bool=None, has_parking_garage: bool=None, search_radius: str=None, has_under_offer_sold_stc: bool=None, has_chain_free: bool=None, has_garden: bool=None, retirement_homes: str=None, auction: str=None, new_homes: str=None, min_price: int=None, min_bedroom: int=None, property_type: str=None, shared_ownership: str=None, max_bedroom: int=None, page: int=None, sort_order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UK property for sale"
    keywords: e.g. conservatory or \"double garage\"
        property_type: Property type comma-separated or empty for all types
Ex: SemiDetached,Flats


`SemiDetached`
`Flats`
`FarmsLand`
`Detached`
`Terraced`
`Bungalows`
`ParkHomes`
        
    """
    url = f"https://zoopla3.p.rapidapi.com/for-sale/property"
    querystring = {'search_area': search_area, }
    if added_to_site:
        querystring['added_to_site'] = added_to_site
    if keywords:
        querystring['keywords'] = keywords
    if has_reduced_price:
        querystring['has_reduced_price'] = has_reduced_price
    if max_price:
        querystring['max_price'] = max_price
    if has_balcony_terrace:
        querystring['has_balcony_terrace'] = has_balcony_terrace
    if has_parking_garage:
        querystring['has_parking_garage'] = has_parking_garage
    if search_radius:
        querystring['search_radius'] = search_radius
    if has_under_offer_sold_stc:
        querystring['has_under_offer_sold_stc'] = has_under_offer_sold_stc
    if has_chain_free:
        querystring['has_chain_free'] = has_chain_free
    if has_garden:
        querystring['has_garden'] = has_garden
    if retirement_homes:
        querystring['retirement_homes'] = retirement_homes
    if auction:
        querystring['auction'] = auction
    if new_homes:
        querystring['new_homes'] = new_homes
    if min_price:
        querystring['min_price'] = min_price
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if property_type:
        querystring['property_type'] = property_type
    if shared_ownership:
        querystring['shared_ownership'] = shared_ownership
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if page:
        querystring['page'] = page
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(search_area: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "auto-complete"
    
    """
    url = f"https://zoopla3.p.rapidapi.com/auto-complete"
    querystring = {'search_area': search_area, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

