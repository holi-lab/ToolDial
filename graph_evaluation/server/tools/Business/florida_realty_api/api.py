import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getlisting(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to get detailed information about a single condominium or townhouse."
    identifier: The unique identifier of a listing you'll got before from getListings.  Each OfferForPurchase has got an attribute named Identifier that is used  on this endpoint to get the details of the property.
        
    """
    url = f"https://florida-realty-api1.p.rapidapi.com/realty/listings/{identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "florida-realty-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimage(filename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Load images of cities you retrieve by calling getCities()."
    filename: The name of the image to load.
        
    """
    url = f"https://florida-realty-api1.p.rapidapi.com/realty/cities/images/{filename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "florida-realty-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of supported cities with some data about those cities like the number of residents, the latitude, longitude, altitude  and a description and image of the city."
    
    """
    url = f"https://florida-realty-api1.p.rapidapi.com/realty/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "florida-realty-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlistings(sort: str='RELEVANCE', offset: int=0, price_max: int=250000, city: str='Cape Coral', limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve a list of condominiums and townhouses in Southwest Florida and on the East Coast."
    sort: Supported values are RELEVANCE (best match), NEWEST (newest first), PRICE_LOW (cheapest first), PRICE_HIGH (cheapest last). By default RELEVANCE is used.
        offset: If you want to implement a paging use offset. The maximum items are defined by the parameter limit starting at offset For example, to return for the frist 10 items set offset to 0 and limit to 10. To return the next 10 items set offset to 10 and limit to 10 and so on. By default offset ist 0.
        price_max: The maximum price of the condos and townhouses you search for in US Dollar. By default 250000 for $250,000 is used.
        city: Name of the city to search for condos and townhouses. Supported cities are Cape Coral, Fort Myers, West Palm Beach, Lake Worth, Lantana, Boynton Beach, Delray Beach, Boca Raton, Pompano Beach, Fort Lauderdale. New supported cities are Naples, Tampa and Saint Petersburg. By default Cape Coral is used.
        limit: The maximum number of items to return in the result list. By default 50 is used.
        
    """
    url = f"https://florida-realty-api1.p.rapidapi.com/realty/listings"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if price_max:
        querystring['price_max'] = price_max
    if city:
        querystring['city'] = city
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "florida-realty-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

