import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_listings(pagesize: int, pagenumber: int=1, orderby: str='ASC', contractaddress: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Top 100 Blur Listings"
    orderby: Listing Price: ASC or DESC
        
    """
    url = f"https://openblur.p.rapidapi.com/listings"
    querystring = {'pageSize': pagesize, }
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if orderby:
        querystring['orderBy'] = orderby
    if contractaddress:
        querystring['contractAddress'] = contractaddress
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openblur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collections(pagesize: int=None, is_from: int=None, slug: str=None, contractaddress: str=None, orderby: str='desc', sortby: str='volumeOneDay', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all Blur Collections"
    orderby: ASC or DESC
        sortby: floorPrice , floorPriceOneDay, floorPriceOneWeek, volumeFifteenMinutes, volumeOneDay, volumeOneWeek, bestCollectionBid, totalCollectionBidValue, totalSupply, numberOwners
        
    """
    url = f"https://openblur.p.rapidapi.com/collections"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if is_from:
        querystring['from'] = is_from
    if slug:
        querystring['slug'] = slug
    if contractaddress:
        querystring['contractAddress'] = contractaddress
    if orderby:
        querystring['orderBy'] = orderby
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openblur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def issuspicious(contractaddress: str=None, afterid: int=None, pagesize: int=50, beforeid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Records a history of isSuspicious / stolen NFTs change over time"
    
    """
    url = f"https://openblur.p.rapidapi.com/isSuspicious"
    querystring = {}
    if contractaddress:
        querystring['contractAddress'] = contractaddress
    if afterid:
        querystring['afterID'] = afterid
    if pagesize:
        querystring['pageSize'] = pagesize
    if beforeid:
        querystring['beforeID'] = beforeid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openblur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_listing_events(pagesize: str, marketplace: str=None, beforeid: str=None, contractaddress: str=None, afterid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Blur Listing Events"
    marketplace: BLUR or OPENSEA
        beforeid: - Descending ID order
        afterid: - Accending ID order
        
    """
    url = f"https://openblur.p.rapidapi.com/listingEvents"
    querystring = {'pageSize': pagesize, }
    if marketplace:
        querystring['marketplace'] = marketplace
    if beforeid:
        querystring['beforeID'] = beforeid
    if contractaddress:
        querystring['contractAddress'] = contractaddress
    if afterid:
        querystring['afterID'] = afterid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openblur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_bids(contractaddress: str, is_from: int=None, take: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve All Blur Bids"
    
    """
    url = f"https://openblur.p.rapidapi.com/bids"
    querystring = {'contractAddress': contractaddress, }
    if is_from:
        querystring['from'] = is_from
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openblur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

