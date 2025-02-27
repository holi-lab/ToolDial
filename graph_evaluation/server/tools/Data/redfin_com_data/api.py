import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(location: str, page: int=None, has_fixer_upper: bool=None, has_waterfront: bool=None, garage_spots: int=None, pool_type: str=None, has_include_outdoor_parking: bool=None, has_elevator: bool=None, has_view: bool=None, min_year_built: int=None, max_year_built: int=None, has_fireplace: bool=None, max_stories: int=None, min_square_feet: int=None, min_lot_size: int=None, time_on_redfin: str=None, has_exclude_55_communities: bool=None, min_stories: int=None, status: str=None, sort: str=None, max_bedroom: int=None, min_bathroom: int=None, max_square_feet: int=None, min_bedroom: int=None, sold_within: str=None, has_air_conditioning: bool=None, has_pets_allowed: bool=None, has_guest_house: bool=None, has_rv_parking: bool=None, keyword_search: str=None, has_green_home: bool=None, has_primary_bedroom_on_main_floor: bool=None, max_lot_size: int=None, has_washer_dryer_hookup: bool=None, home_type: str=None, min_price: int=None, max_price: int=None, search_by: str=None, sub_location: str=None, search_type: str=None, has_accessible_home: bool=None, has_basement: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    location: Address or zip code
        garage_spots: garage_spots: 1->5
        sold_within: This option only has an available when `search_type=Sold`
Default:` Last-3Months`
        home_type: Property type comma-separated or empty for all types:

```
House
Townhouse
Condo
Land
MultiFamily
Mobile
Co-op
Other
```
Ex: House,Townhouse
 
        search_by: Default: places 
        search_type: Default: `ForSale`
        
    """
    url = f"https://redfin-com-data.p.rapidapi.com/property/search"
    querystring = {'location': location, }
    if page:
        querystring['page'] = page
    if has_fixer_upper:
        querystring['has_fixer_upper'] = has_fixer_upper
    if has_waterfront:
        querystring['has_waterfront'] = has_waterfront
    if garage_spots:
        querystring['garage_spots'] = garage_spots
    if pool_type:
        querystring['pool_type'] = pool_type
    if has_include_outdoor_parking:
        querystring['has_include_outdoor_parking'] = has_include_outdoor_parking
    if has_elevator:
        querystring['has_elevator'] = has_elevator
    if has_view:
        querystring['has_view'] = has_view
    if min_year_built:
        querystring['min_year_built'] = min_year_built
    if max_year_built:
        querystring['max_year_built'] = max_year_built
    if has_fireplace:
        querystring['has_fireplace'] = has_fireplace
    if max_stories:
        querystring['max_stories'] = max_stories
    if min_square_feet:
        querystring['min_square_feet'] = min_square_feet
    if min_lot_size:
        querystring['min_lot_size'] = min_lot_size
    if time_on_redfin:
        querystring['time_on_redfin'] = time_on_redfin
    if has_exclude_55_communities:
        querystring['has_exclude_55_communities'] = has_exclude_55_communities
    if min_stories:
        querystring['min_stories'] = min_stories
    if status:
        querystring['status'] = status
    if sort:
        querystring['sort'] = sort
    if max_bedroom:
        querystring['max_bedroom'] = max_bedroom
    if min_bathroom:
        querystring['min_bathroom'] = min_bathroom
    if max_square_feet:
        querystring['max_square_feet'] = max_square_feet
    if min_bedroom:
        querystring['min_bedroom'] = min_bedroom
    if sold_within:
        querystring['sold_within'] = sold_within
    if has_air_conditioning:
        querystring['has_air_conditioning'] = has_air_conditioning
    if has_pets_allowed:
        querystring['has_pets_allowed'] = has_pets_allowed
    if has_guest_house:
        querystring['has_guest_house'] = has_guest_house
    if has_rv_parking:
        querystring['has_rv_parking'] = has_rv_parking
    if keyword_search:
        querystring['keyword_search'] = keyword_search
    if has_green_home:
        querystring['has_green_home'] = has_green_home
    if has_primary_bedroom_on_main_floor:
        querystring['has_primary_bedroom_on_main_floor'] = has_primary_bedroom_on_main_floor
    if max_lot_size:
        querystring['max_lot_size'] = max_lot_size
    if has_washer_dryer_hookup:
        querystring['has_washer_dryer_hookup'] = has_washer_dryer_hookup
    if home_type:
        querystring['home_type'] = home_type
    if min_price:
        querystring['min_price'] = min_price
    if max_price:
        querystring['max_price'] = max_price
    if search_by:
        querystring['search_by'] = search_by
    if sub_location:
        querystring['sub_location'] = sub_location
    if search_type:
        querystring['search_type'] = search_type
    if has_accessible_home:
        querystring['has_accessible_home'] = has_accessible_home
    if has_basement:
        querystring['has_basement'] = has_basement
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete"
    
    """
    url = f"https://redfin-com-data.p.rapidapi.com/property/auto-complete"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property details"
    url: **1. The value of the url in  the  /Search endpoint**
Ex: /NY/Jamaica/9452-199th-St-11423/home/20743109
**2. Or copy the URL from the browser**
Ex: https://www.redfin.com/NY/Jamaica/9452-199th-St-11423/home/20743109 
        
    """
    url = f"https://redfin-com-data.p.rapidapi.com/property/detail"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by URL"
    
    """
    url = f"https://redfin-com-data.p.rapidapi.com/property/search-url"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_detail(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "agents/detail"
    
    """
    url = f"https://redfin-com-data.p.rapidapi.com/agents/detail"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_search(location: str, sub_location: str='T B, MD, USA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "agents/search"
    location: Neighborhood, City, ZIP, Agent name
        
    """
    url = f"https://redfin-com-data.p.rapidapi.com/agents/search"
    querystring = {'location': location, }
    if sub_location:
        querystring['sub_location'] = sub_location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agents_auto_complete(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "agents/auto-complete"
    
    """
    url = f"https://redfin-com-data.p.rapidapi.com/agents/auto-complete"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redfin-com-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

