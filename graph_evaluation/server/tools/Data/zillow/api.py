import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def walk_transit_and_bike_scores(zpid: str='20485700', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Walk, Transit and Bike Scores of a property by zpid"
    
    """
    url = f"https://zillow69.p.rapidapi.com/walkTransitBikeScores"
    querystring = {}
    if zpid:
        querystring['zpid'] = zpid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow69.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_details(zpid: int=None, property_url: str='https://www.zillow.com/homedetails/101-California-Ave-UNIT-303-Santa-Monica-CA-90403/20485700_zpid/', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Property details"
    
    """
    url = f"https://zillow69.p.rapidapi.com/property"
    querystring = {}
    if zpid:
        querystring['zpid'] = zpid
    if property_url:
        querystring['property_url'] = property_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow69.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agent_s_active_listings(zuid: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Agent's active listings"
    
    """
    url = f"https://zillow69.p.rapidapi.com/agent/listings"
    querystring = {'zuid': zuid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow69.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_agent(page: str=None, locationtext: str='Newport Beach', name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Agent"
    locationtext: e.g. `Newport Beach` or zip code `90278`
        
    """
    url = f"https://zillow69.p.rapidapi.com/agent/search"
    querystring = {}
    if page:
        querystring['page'] = page
    if locationtext:
        querystring['locationText'] = locationtext
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow69.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_properties_by_url(url: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of properties by providing the zillow search results URL"
    
    """
    url = f"https://zillow69.p.rapidapi.com/searchByUrl"
    querystring = {'url': url, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow69.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_properties(location: str, iscomingsoon: int=None, buildyearmax: int=None, isbasementfinished: int=None, bedsmin: int=None, bathsmin: int=None, bathsmax: int=None, ispendingundercontract: int=None, sort: str=None, sqftmax: int=None, home_type: str=None, rentminprice: int=None, isnewconstruction: int=None, keywords: str=None, otherlistings: int=None, isbasementunfinished: int=None, dayson: str=None, sqftmin: int=None, buildyearmin: int=None, soldinlast: str=None, rentmaxprice: int=None, bedsmax: int=None, minprice: int=None, page: int=None, maxprice: int=None, status_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for properties"
    location: Location details, address, county, neighborhood or Zip code.
        sort: For `status_type` = `ForSale` OR `RecentlySold` are available:
- `Homes_for_You`
- `Price_High_Low`
- `Price_Low_High`
- `Newest`
- `Bedrooms`
- `Bathrooms`
- `Square_Feet`
- `Lot_Size`

default `Homes_for_You`

For `status_type` = `ForRent` are available:
- `Verified_Source`
- `Payment_High_Low`
- `Payment_Low_High`
- `Newest`
- `Bedrooms`
- `Bathrooms`
- `Square_Feet`
- `Lot_Size`

default `Verified_Source`
        home_type: Property type comma-separated or empty for all types:
- `Multi-family`
- `Apartments`
- `Houses`
- `Manufactured`
- `Condos`
- `LotsLand`
- `Townhomes`
        keywords: MLS #, yard, etc.
        otherlistings: If set to 1, the results will only include data from the `Other Listings` tab.
        dayson: Days on Zillow possible values :
- empty: Any,
- 1: 1 day,
- 7: 7 days,
- 14: 14 days,
- 30: 30 days,
- 90: 90 days,
- 6m: 6 months,
- 12m: 12 months,
- 24m: 24 months,
- 36m: 36 months
        soldinlast: Sold In Last
        page: Page number if at the previous response `totalPages` > 1
        status_type: Status type of the properties

Default : forSale

- forSale

- forRent

- recentlySold

        
    """
    url = f"https://zillow69.p.rapidapi.com/search"
    querystring = {'location': location, }
    if iscomingsoon:
        querystring['isComingSoon'] = iscomingsoon
    if buildyearmax:
        querystring['buildYearMax'] = buildyearmax
    if isbasementfinished:
        querystring['isBasementFinished'] = isbasementfinished
    if bedsmin:
        querystring['bedsMin'] = bedsmin
    if bathsmin:
        querystring['bathsMin'] = bathsmin
    if bathsmax:
        querystring['bathsMax'] = bathsmax
    if ispendingundercontract:
        querystring['isPendingUnderContract'] = ispendingundercontract
    if sort:
        querystring['sort'] = sort
    if sqftmax:
        querystring['sqftMax'] = sqftmax
    if home_type:
        querystring['home_type'] = home_type
    if rentminprice:
        querystring['rentMinPrice'] = rentminprice
    if isnewconstruction:
        querystring['isNewConstruction'] = isnewconstruction
    if keywords:
        querystring['keywords'] = keywords
    if otherlistings:
        querystring['otherListings'] = otherlistings
    if isbasementunfinished:
        querystring['isBasementUnfinished'] = isbasementunfinished
    if dayson:
        querystring['daysOn'] = dayson
    if sqftmin:
        querystring['sqftMin'] = sqftmin
    if buildyearmin:
        querystring['buildYearMin'] = buildyearmin
    if soldinlast:
        querystring['soldInLast'] = soldinlast
    if rentmaxprice:
        querystring['rentMaxPrice'] = rentmaxprice
    if bedsmax:
        querystring['bedsMax'] = bedsmax
    if minprice:
        querystring['minPrice'] = minprice
    if page:
        querystring['page'] = page
    if maxprice:
        querystring['maxPrice'] = maxprice
    if status_type:
        querystring['status_type'] = status_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow69.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

