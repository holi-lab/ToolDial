import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def monthly_volume_by_gallery(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of monthly volume by gallery"
    
    """
    url = f"https://most-expensive-nft-artworks.p.rapidapi.com/monthly_crypto_art_volume"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "most-expensive-nft-artworks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_sold(page: str='1', gallery: str=None, sort: str='date_sold', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent sold artworks list.
		
		**Possible filtering by gallery: **
		Art Blocks
		Async Art
		Auction Houses
		Foundation
		hic et nunc
		KnownOrigin
		MakersPlace
		Nifty Gateway
		SuperRare"
    
    """
    url = f"https://most-expensive-nft-artworks.p.rapidapi.com/artworks"
    querystring = {}
    if page:
        querystring['page'] = page
    if gallery:
        querystring['gallery'] = gallery
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "most-expensive-nft-artworks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_expensive(sort: str='usd_price', page: str='1', gallery: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most expensive artworks list.
		
		**Possible filtering by gallery: **
		Art Blocks
		Async Art
		Auction Houses
		Foundation
		hic et nunc
		KnownOrigin
		MakersPlace
		Nifty Gateway
		SuperRare"
    
    """
    url = f"https://most-expensive-nft-artworks.p.rapidapi.com/artworks"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if gallery:
        querystring['gallery'] = gallery
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "most-expensive-nft-artworks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

