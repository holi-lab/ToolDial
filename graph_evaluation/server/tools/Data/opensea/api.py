import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def collections(limit: int=300, asset_owner: str=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/collections/` endpoint provides a list of all the collections supported and vetted by OpenSea. To include all collections relevant to a user (including non-whitelisted ones), set the` owner` param.
		
		Each collection in the returned area has an attribute called `primary_asset_contracts` with info about the smart contracts belonging to that collection. For example, ERC-1155 contracts maybe have multiple collections all referencing the same contract, but many ERC-721 contracts may all belong to the same collection (dapp).
		
		You can also use this endpoint to find which dapps an account uses, and how many items they own in each - all in one API call!"
    limit: For pagination. Maximum number of contracts to return.
        asset_owner: A wallet address. If specified, will return collections where the owner owns at least one asset belonging to smart contracts in the collection. The number of assets the account owns is shown as `owned_asset_count` for each collection.
        offset: For pagination. Number of contracts offset from the beginning of the result list.
        
    """
    url = f"https://opensea13.p.rapidapi.com/collections/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if asset_owner:
        querystring['asset_owner'] = asset_owner
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def collection_stats(collection_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to fetch stats for a specific collection, including realtime floor price statistics."
    collection_slug: The collection slug
        
    """
    url = f"https://opensea13.p.rapidapi.com/collection_stats/"
    querystring = {'collection_slug': collection_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asset_owners(asset_contract_address: str, token_id: str, cursor: str=None, order_direction: str='desc', limit: int=20, order_by: str='created_date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to obtain the entire list of owners for an NFT. Results will also include the quantity owned."
    asset_contract_address: Address of the contract for this NFT
        token_id: Token ID for this item
        cursor: The cursor token. Used for going forward / backward to the next/previous pages.
        order_direction: asc or desc. Descending is the default.
        limit: How many results to show. The default value is 20 (if param is omitted), max is 50.
        order_by: What field to order by. For now, we only support `created_date`.
        
    """
    url = f"https://opensea13.p.rapidapi.com/asset_owners/"
    querystring = {'asset_contract_address': asset_contract_address, 'token_id': token_id, }
    if cursor:
        querystring['cursor'] = cursor
    if order_direction:
        querystring['order_direction'] = order_direction
    if limit:
        querystring['limit'] = limit
    if order_by:
        querystring['order_by'] = order_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_contract(asset_contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to fetch more in-depth information about an contract asset."
    asset_contract_address: Address of the contract
        
    """
    url = f"https://opensea13.p.rapidapi.com/asset_contract/"
    querystring = {'asset_contract_address': asset_contract_address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def assets(asset_contract_addresses: str=None, order_direction: str='desc', include_orders: bool=None, limit: int=20, collection_slug: str='cryptopunks', cursor: str=None, collection_editor: str=None, asset_contract_address: str=None, owner: str=None, collection: str=None, token_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To retrieve assets from our API, call the `/assets/` endpoint with the desired filter parameters."
    asset_contract_addresses: An array of contract addresses to search for (e.g. `?asset_contract_addresses=0x1...&asset_contract_addresses=0x2...`). Will return a list of assets with contracts matching any of the addresses in this array. If **token_ids** is also specified, then it will only return assets that match each `(address, token_id)` pairing, respecting order.
        order_direction: Can be `asc` for ascending or `desc` for descending
        include_orders: A flag determining if order information should be included in the response.
        limit: Limit. Defaults to 20, capped at 50.
        collection_slug: Limit responses to members of a collection. Case sensitive and must match the collection slug exactly. Will return all assets from all contracts in a collection.
        cursor: A cursor pointing to the page to retrieve
        asset_contract_address: The NFT contract address for the assets
        collection: Limit responses to members of a collection. Case sensitive and must match the collection slug exactly. Will return all assets from all contracts in a collection.
        token_ids: An array of token IDs to search for (e.g. `?token_ids=1&token_ids=209`). Will return a list of assets with token_id matching any of the IDs in this array.
        
    """
    url = f"https://opensea13.p.rapidapi.com/assets/"
    querystring = {}
    if asset_contract_addresses:
        querystring['asset_contract_addresses'] = asset_contract_addresses
    if order_direction:
        querystring['order_direction'] = order_direction
    if include_orders:
        querystring['include_orders'] = include_orders
    if limit:
        querystring['limit'] = limit
    if collection_slug:
        querystring['collection_slug'] = collection_slug
    if cursor:
        querystring['cursor'] = cursor
    if collection_editor:
        querystring['collection_editor'] = collection_editor
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if owner:
        querystring['owner'] = owner
    if collection:
        querystring['collection'] = collection
    if token_ids:
        querystring['token_ids'] = token_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_collection(collection_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used for retrieving more in-depth information about individual collections, including real time statistics like floor price."
    collection_slug: The collection slug of this collection that is used to uniquely link to this collection on OpenSea
        
    """
    url = f"https://opensea13.p.rapidapi.com/collection/"
    querystring = {'collection_slug': collection_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bundles(asset_contract_address: str=None, on_sale: bool=None, owner: str=None, limit: int=20, offset: int=0, asset_contract_addresses: str=None, token_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bundles are groups of items for sale on OpenSea. You can buy them all at once in one transaction, and you can create them without any transactions or gas, as long as you've already approved the assets inside."
    asset_contract_address: Contract address of all the assets in a bundle. Only works for homogenous bundles, not mixed ones.
        on_sale: If set to `true`, only show bundles currently on sale. If set to `false`, only show bundles that have been sold or cancelled.
        owner: Account address of the owner of a bundle (and all assets within)
        limit: For pagination: how many results to return
        offset: For pagination: the index of the result to start at (beginning with 0)
        asset_contract_addresses: An array of contract addresses to search for (e.g. `?asset_contract_addresses=0x1...&asset_contract_addresses=0x2...`). Will return a list of bundles with assets having contracts that match ALL of the addresses in this array. Useful for querying for mixed bundles, e.g. ones with CryptoKitties AND CK Talisman statues.
        token_ids: A list of token IDs for showing only bundles with at least one of the token IDs in the list
        
    """
    url = f"https://opensea13.p.rapidapi.com/bundles/"
    querystring = {}
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if on_sale:
        querystring['on_sale'] = on_sale
    if owner:
        querystring['owner'] = owner
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if asset_contract_addresses:
        querystring['asset_contract_addresses'] = asset_contract_addresses
    if token_ids:
        querystring['token_ids'] = token_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seaport_offers(asset_contract_address: str=None, limit: str=None, token_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch the set of active offers on a given NFT for the Seaport contract."
    asset_contract_address: Address of the contract for an NFT
        limit: Number of offers to retrieve
        token_ids: An array of token IDs to search for (e.g. `?token_ids=1&token_ids=209`). This endpoint will return a list of offers with token_id matching any of the IDs in this array.
        
    """
    url = f"https://opensea13.p.rapidapi.com/v2/orders/ethereum/seaport/offers"
    querystring = {}
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if limit:
        querystring['limit'] = limit
    if token_ids:
        querystring['token_ids'] = token_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seaport_listings(token_ids: str=None, asset_contract_address: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch the set of active listings on a given NFT for the Seaport contract."
    token_ids: An array of token IDs to search for (e.g. `?token_ids=1&token_ids=209`). This endpoint will return a list of listings with token_id matching any of the IDs in this array.
        asset_contract_address: Address of the contract for an NFT
        limit: Number of listings to retrieve
        
    """
    url = f"https://opensea13.p.rapidapi.com/v2/orders/ethereum/seaport/listings"
    querystring = {}
    if token_ids:
        querystring['token_ids'] = token_ids
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

