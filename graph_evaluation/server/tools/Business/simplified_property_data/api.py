import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(location: str, min_sqft: int=None, min_yearbuilt: int=None, max_yearbuilt: int=None, max_sqft: int=None, max_lotsqft: int=None, min_lotsqft: int=None, min_price: int=None, min_baths: int=None, max_price: int=None, max_maths: int=None, has_pool: bool=None, newconstruction: bool=None, has_ac: bool=None, salebyagent: bool=None, waterfront: bool=None, exclude55plus: bool=None, comingsoon: bool=None, singlestory: bool=None, keywords: str=None, days: str=None, view: str='park,water', saleauction: bool=None, salebyowner: bool=None, foreclosure: bool=None, status: str=None, page: int=None, type: str='houses,condos', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of properties based on the search criteria."
    min_price: Minimum price of the listing

if `status` is `ForRent` this is the minimum monthly rent payment.
        max_price: Maximum price of the listing

if `status` is `ForRent` this is the maximum monthly rent payment.
        has_pool: Default: `false`

Whether to only show properties that have a pool
        newconstruction: Default: `true`

Determines if properties that are new construction should be returned
does nothing if `status` isn't `ForSale`
        has_ac: Default: `false`

Whether to only show properties that have air conditioning
        salebyagent: Default: `true`

If properties sold by an agent should be returned
does nothing if `status` isn't `ForSale`
        waterfront: Default: `false`

Whether to only show properties that waterfront
        exclude55plus: Default: `false`

If true, communities for people aged 55+ will not be returned.
        comingsoon: Default: `true`

Shows properties that will be entering the market soon.
does nothing if `status` isn't `ForSale`
        singlestory: Default: `false`

Show only single story properties
        days: Days the property has been listed If `status` is `ForRent` or `ForSale`. 
If status is `RecentlySold`, then it is how many days ago it was sold
        view: Comma separated list of views from property

Possible values:
* city
* park
* water
* mountain
        saleauction: Default: `true`

Determines if auctions should be returned
does nothing if `status` isn't `ForSale`
        salebyowner: Default: `true`

Determines if properties sold by owner should be returned
does nothing if `status` isn't `ForSale`
        foreclosure: Default: `true`

Whether to show propertiers in foreclosure or not
does nothing if `status` isn't `ForSale`
        status: Default: `ForSale`
        type: Comma separated list of housing types to return.

Possible values:

if `status` is `ForSale` or `RecentlySold`
* houses
* townhouses
* apartments
* condos
* multifamily
* manufactured
* lots

if `status` is `ForRent`:
* apartments_condos_coops
* houses
* townhouses
        
    """
    url = f"https://simplified-property-data.p.rapidapi.com/search"
    querystring = {'location': location, }
    if min_sqft:
        querystring['min_sqft'] = min_sqft
    if min_yearbuilt:
        querystring['min_yearbuilt'] = min_yearbuilt
    if max_yearbuilt:
        querystring['max_yearbuilt'] = max_yearbuilt
    if max_sqft:
        querystring['max_sqft'] = max_sqft
    if max_lotsqft:
        querystring['max_lotsqft'] = max_lotsqft
    if min_lotsqft:
        querystring['min_lotsqft'] = min_lotsqft
    if min_price:
        querystring['min_price'] = min_price
    if min_baths:
        querystring['min_baths'] = min_baths
    if max_price:
        querystring['max_price'] = max_price
    if max_maths:
        querystring['max_maths'] = max_maths
    if has_pool:
        querystring['has_pool'] = has_pool
    if newconstruction:
        querystring['newconstruction'] = newconstruction
    if has_ac:
        querystring['has_ac'] = has_ac
    if salebyagent:
        querystring['salebyagent'] = salebyagent
    if waterfront:
        querystring['waterfront'] = waterfront
    if exclude55plus:
        querystring['exclude55plus'] = exclude55plus
    if comingsoon:
        querystring['comingsoon'] = comingsoon
    if singlestory:
        querystring['singlestory'] = singlestory
    if keywords:
        querystring['keywords'] = keywords
    if days:
        querystring['days'] = days
    if view:
        querystring['view'] = view
    if saleauction:
        querystring['saleauction'] = saleauction
    if salebyowner:
        querystring['salebyowner'] = salebyowner
    if foreclosure:
        querystring['foreclosure'] = foreclosure
    if status:
        querystring['status'] = status
    if page:
        querystring['page'] = page
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simplified-property-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_lookup(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets detailed information about a property listing"
    
    """
    url = f"https://simplified-property-data.p.rapidapi.com/property"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simplified-property-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

