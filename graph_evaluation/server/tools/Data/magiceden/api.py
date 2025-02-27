import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def buy(buyerpublickey: str=None, buyeraddress: str='bc1qu456w6gka2aug8luya94sn9grkkdv5m0avua6d', buyertokenreceiveaddress: str='bc1qu456w6gka2aug8luya94sn9grkkdv5m0avua6d', feeratetier: str='halfHourFee', tokenid: str='0c265fd549c91f1ee1d6e07e8d3061c80ccb71c58295e0934fd0e140b91c8b26i0', price: int=31000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Buy"
    
    """
    url = f"https://magiceden1.p.rapidapi.com/v2/ord/btc/psbt/buying"
    querystring = {}
    if buyerpublickey:
        querystring['buyerPublicKey'] = buyerpublickey
    if buyeraddress:
        querystring['buyerAddress'] = buyeraddress
    if buyertokenreceiveaddress:
        querystring['buyerTokenReceiveAddress'] = buyertokenreceiveaddress
    if feeratetier:
        querystring['feerateTier'] = feeratetier
    if tokenid:
        querystring['tokenId'] = tokenid
    if price:
        querystring['price'] = price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magiceden1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_activities_btc(kind: str='buying_broadcasted', collectionsymbol: str='omb', limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve activities (BTC)"
    
    """
    url = f"https://magiceden1.p.rapidapi.com/v2/ord/btc/activities"
    querystring = {}
    if kind:
        querystring['kind[]'] = kind
    if collectionsymbol:
        querystring['collectionSymbol'] = collectionsymbol
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magiceden1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_tokens_matic(slug: str, sort: str='askAmountNum', limit: int=20, marketplacefilter: str='["blur","looksrare","magiceden_opensea","magiceden_zero_ex","opensea","opensea_wyvern_v1","opensea_wyvern_v2","rarible","x2y2"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://polygon-api.magiceden.io"
    
    """
    url = f"https://magiceden1.p.rapidapi.com/v2/xc/collections/polygon/{slug}/orders"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if limit:
        querystring['limit'] = limit
    if marketplacefilter:
        querystring['marketplaceFilter'] = marketplacefilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magiceden1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_tokens_sol(collectionsymbol: str='smb_gen3', agg: int=3, mode: str='all', field: int=1, limit: int=40, direction: int=2, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve tokens (SOL)"
    
    """
    url = f"https://magiceden1.p.rapidapi.com/idxv2/getListedNftsByCollectionSymbol"
    querystring = {}
    if collectionsymbol:
        querystring['collectionSymbol'] = collectionsymbol
    if agg:
        querystring['agg'] = agg
    if mode:
        querystring['mode'] = mode
    if field:
        querystring['field'] = field
    if limit:
        querystring['limit'] = limit
    if direction:
        querystring['direction'] = direction
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magiceden1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_tokens_eth(slug: str, marketplacefilter: str='["magiceden_seaport","blur"]', enableblurlistings: bool=None, limit: int=20, sort: str='askAmountNum', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve tokens"
    
    """
    url = f"https://magiceden1.p.rapidapi.com/v2/xc/collections/eth/{slug}/orders"
    querystring = {}
    if marketplacefilter:
        querystring['marketplaceFilter'] = marketplacefilter
    if enableblurlistings:
        querystring['enableBlurListings'] = enableblurlistings
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magiceden1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_tokens_btc(sortby: str='priceAsc', collectionsymbol: str='omb', maxprice: int=0, disablependingtransactions: bool=None, limit: int=20, offset: int=0, minprice: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve tokens (BTC)"
    
    """
    url = f"https://magiceden1.p.rapidapi.com/v2/ord/btc/tokens"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if collectionsymbol:
        querystring['collectionSymbol'] = collectionsymbol
    if maxprice:
        querystring['maxPrice'] = maxprice
    if disablependingtransactions:
        querystring['disablePendingTransactions'] = disablependingtransactions
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if minprice:
        querystring['minPrice'] = minprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magiceden1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the full Magic Eden URL to bypass in the header request"
    
    """
    url = f"https://magiceden1.p.rapidapi.com/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magiceden1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

