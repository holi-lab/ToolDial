import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def activities_transfers(chain_id: str, collection_id: str, is_whale: str='false', order_by: str='-time_at', limit: str='20', start: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get collection activity**
		
		**Pagination example**:
		 start=0&limit=20
		
		**Supported chain_id**: 
		eth, bsc, matic, avax, arb, op
		
		**order_by** 
		-time_at
		time_at
		usd_value
		-usd_value"
    
    """
    url = f"https://rich-nft-api-metadata.p.rapidapi.com/activity/list"
    querystring = {'chain_id': chain_id, 'collection_id': collection_id, }
    if is_whale:
        querystring['is_whale'] = is_whale
    if order_by:
        querystring['order_by'] = order_by
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rich-nft-api-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_holders(chain_id: str, is_id: str, limit: str='20', start: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top holders 
		
		**Pagination example**:
		 start=0&limit=20
		
		**Supported chain_id**: 
		eth, bsc, matic, avax, arb, op"
    
    """
    url = f"https://rich-nft-api-metadata.p.rapidapi.com/collection/top_holders"
    querystring = {'chain_id': chain_id, 'id': is_id, }
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rich-nft-api-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_traits(is_id: str, chain_id: str, start: str='0', order_by: str='-rarity_rank_at', limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get NFT tokens in specific collection.
		
		**Pagination example**:
		 start=0&limit=20
		
		**Supported chain_id**: 
		eth, bsc, matic, avax, arb, op
		
		**Filter by traits example**:
		traits=[{"name":"type","value":"Female"}]
		
		**order_by** 
		rarity_rank_at 
		OR
		-rarity_rank_at"
    
    """
    url = f"https://rich-nft-api-metadata.p.rapidapi.com/collection/top_traits"
    querystring = {'id': is_id, 'chain_id': chain_id, }
    if start:
        querystring['start'] = start
    if order_by:
        querystring['order_by'] = order_by
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rich-nft-api-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traits_for_collection(chain_id: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all NFT traits attributes of a specific collection
		
		**Supported chain_id**: 
		eth, bsc, matic, avax, arb, op"
    
    """
    url = f"https://rich-nft-api-metadata.p.rapidapi.com/collection/traits/summary"
    querystring = {'chain_id': chain_id, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rich-nft-api-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nft_tokens_in_collection(is_id: str, chain_id: str, traits: str='[{"name":"type","value":"Female"}]', order_by: str='-value', start: str='0', limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get NFT tokens in specific collection.
		
		**Pagination example**:
		 start=0&limit=20
		
		**Supported chain_id**: 
		eth, bsc, matic, avax, arb, op
		
		**Filter by traits example**:
		traits=[{"name":"type","value":"Female"}]
		
		**order_by: options** 
		value
		-value
		rarity_rank_at
		-rarity_rank_at
		pay_usd_value
		-pay_usd_value"
    
    """
    url = f"https://rich-nft-api-metadata.p.rapidapi.com/collection/nft_list"
    querystring = {'id': is_id, 'chain_id': chain_id, }
    if traits:
        querystring['traits'] = traits
    if order_by:
        querystring['order_by'] = order_by
    if start:
        querystring['start'] = start
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rich-nft-api-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def collection_info(is_id: str, chain_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info about collection.
		Supported chain_id: eth, bsc, matic, avax, arb, op"
    
    """
    url = f"https://rich-nft-api-metadata.p.rapidapi.com/collection"
    querystring = {'id': is_id, 'chain_id': chain_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rich-nft-api-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def collections(chain_id: str, limit: str='20', start: str='0', q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get NFT collections.
		
		Pagination example: start=0&limit=100
		Search for collections: q=CRYPTOPUNKS
		Supported chain ids: eth, bsc, matic, avax, arb, op"
    
    """
    url = f"https://rich-nft-api-metadata.p.rapidapi.com/collection/list"
    querystring = {'chain_id': chain_id, }
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rich-nft-api-metadata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

