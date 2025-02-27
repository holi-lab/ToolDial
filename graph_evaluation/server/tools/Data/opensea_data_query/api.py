import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def deprecated_retrieving_offers(asset_contract_address: str, token_id: str, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to fetch active offers for a given asset."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/asset/{asset_contract_address}/{token_id}/offers"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deprecated_retrieving_listings(asset_contract_address: str, token_id: str, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to fetch active listings on a given asset."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/asset/{asset_contract_address}/{token_id}/listings"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_trait_offers_v2(slug: str, type: str='tier', value: str='classy_1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get all trait offers for a specified collection."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/offers/collection/{slug}/traits"
    querystring = {}
    if type:
        querystring['type'] = type
    if value:
        querystring['value'] = value
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_offers_v2(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get collection offers for a specified collection."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/offers/collection/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_offers_v2(chain: str, asset_contract_address: str='0x4372f4d950d30c6f12c7228ade77d6cc019404c9', limit: str='20', token_ids: str='309', taker: str=None, maker: str=None, listed_after: str=None, order_direction: str=None, order_by: str=None, listed_before: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch the set of active offers on a given NFT for the Seaport contract."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/orders/{chain}/seaport/offers"
    querystring = {}
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if limit:
        querystring['limit'] = limit
    if token_ids:
        querystring['token_ids'] = token_ids
    if taker:
        querystring['taker'] = taker
    if maker:
        querystring['maker'] = maker
    if listed_after:
        querystring['listed_after'] = listed_after
    if order_direction:
        querystring['order_direction'] = order_direction
    if order_by:
        querystring['order_by'] = order_by
    if listed_before:
        querystring['listed_before'] = listed_before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_listings_v2(slug: str, limit: int=100, next: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all active listings for a collection"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/listings/collection/{slug}/all"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_nfts_by_collection(slug: str, limit: str='20', next: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch multiple NFTs for a collection"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/collection/{slug}/nfts"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_an_nft(identifier: str, chain: str, address: str, limit: str='20', next: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch metadata, traits, ownership information, and rarity for an NFT"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/chain/{chain}/contract/{address}/nfts/{identifier}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_nfts_by_contract(chain: str, address: str, next: str=None, limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch multiple NFTs for a smart contract"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/chain/{chain}/contract/{address}/nfts"
    querystring = {}
    if next:
        querystring['next'] = next
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_nfts_by_account(chain: str, address: str, next: str=None, limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch NFTs owned by a given account address"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/chain/{chain}/account/{address}/nfts"
    querystring = {}
    if next:
        querystring['next'] = next
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_offers_v2(slug: str, next: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all open offers for a collection"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/offers/collection/{slug}/all"
    querystring = {}
    if next:
        querystring['next'] = next
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_listings_v2(chain: str, maker: str=None, limit: str='20', asset_contract_address: str='0x4372f4d950d30c6f12c7228ade77d6cc019404c9', token_ids: str='309', listed_before: str=None, order_by: str=None, order_direction: str=None, listed_after: str=None, taker: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch the set of active listings on a given NFT for the Seaport contract."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/v2/orders/{chain}/seaport/listings"
    querystring = {}
    if maker:
        querystring['maker'] = maker
    if limit:
        querystring['limit'] = limit
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if token_ids:
        querystring['token_ids'] = token_ids
    if listed_before:
        querystring['listed_before'] = listed_before
    if order_by:
        querystring['order_by'] = order_by
    if order_direction:
        querystring['order_direction'] = order_direction
    if listed_after:
        querystring['listed_after'] = listed_after
    if taker:
        querystring['taker'] = taker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_a_collection(collection_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used for retrieving more in-depth information about individual collections, including real time statistics like floor price."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/collection/{collection_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_owners(token_id: str, asset_contract_address: str, order_by: str=None, cursor: str=None, limit: int=None, order_direction: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to obtain the entire list of owners for an NFT. Results will also include the quantity owned."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/asset/{asset_contract_address}/{token_id}/owners"
    querystring = {}
    if order_by:
        querystring['order_by'] = order_by
    if cursor:
        querystring['cursor'] = cursor
    if limit:
        querystring['limit'] = limit
    if order_direction:
        querystring['order_direction'] = order_direction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_collection_stats(collection_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to fetch stats for a specific collection, including realtime floor price statistics"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/collection/{collection_slug}/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_bundles(asset_contract_addresses: str=None, limit: int=1, token_ids: int=None, on_sale: bool=None, asset_contract_address: str=None, offset: int=0, owner: str='0x842858c0093866abd09a363150fb540d97e78223', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bundles are groups of items for sale on OpenSea. You can buy them all at once in one transaction, and you can create them without any transactions or gas, as long as you've already approved the assets inside."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/bundles"
    querystring = {}
    if asset_contract_addresses:
        querystring['asset_contract_addresses'] = asset_contract_addresses
    if limit:
        querystring['limit'] = limit
    if token_ids:
        querystring['token_ids'] = token_ids
    if on_sale:
        querystring['on_sale'] = on_sale
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if offset:
        querystring['offset'] = offset
    if owner:
        querystring['owner'] = owner
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_an_asset(token_id: str, asset_contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you're having issues getting your items to show up properly on OpenSea (perhaps they're missing an image or attributes), you can debug your metadata using the /validate API endpoint."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/asset/{asset_contract_address}/{token_id}/validate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_collections(limit: int=20, asset_owner: str='0x2bf699087a0d1d67519ba86f960fecd80d59c4d7', offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/collections` endpoint provides a list of all the collections supported and vetted by OpenSea. To include all collections relevant to a user (including non-whitelisted ones), set the owner param.
		
		Each collection in the returned area has an attribute called primary_asset_contracts with info about the smart contracts belonging to that collection. For example, ERC-1155 contracts maybe have multiple collections all referencing the same contract, but many ERC-721 contracts may all belong to the same collection (dapp).
		
		You can also use this endpoint to find which dapps an account uses, and how many items they own in each - all in one API call!"
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/collections"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if asset_owner:
        querystring['asset_owner'] = asset_owner
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_assets(owner: str=None, order_direction: str='desc', asset_contract_address: str=None, limit: int=20, collection_slug: str=None, cursor: str=None, token_ids: int=None, asset_contract_addresses: str=None, collection: str='ongakucraft', include_orders: bool=None, collection_editor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To retrieve assets from our API, call the `/assets` endpoint with the desired filter parameters.
		
		Auctions created on OpenSea don't use an escrow contract, which enables gas-free auctions and allows users to retain ownership of their items while they're on sale. So this is just a heads up in case you notice some assets from opensea.io not appearing in the API."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/assets"
    querystring = {}
    if owner:
        querystring['owner'] = owner
    if order_direction:
        querystring['order_direction'] = order_direction
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if limit:
        querystring['limit'] = limit
    if collection_slug:
        querystring['collection_slug'] = collection_slug
    if cursor:
        querystring['cursor'] = cursor
    if token_ids:
        querystring['token_ids'] = token_ids
    if asset_contract_addresses:
        querystring['asset_contract_addresses'] = asset_contract_addresses
    if collection:
        querystring['collection'] = collection
    if include_orders:
        querystring['include_orders'] = include_orders
    if collection_editor:
        querystring['collection_editor'] = collection_editor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_a_contract(asset_contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to fetch more in-depth information about an contract asset."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/asset_contract/{asset_contract_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_an_asset(token_id: str, asset_contract_address: str, include_orders: bool=None, account_address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to fetch more in-depth information about an individual asset.
		
		To retrieve an individual from our API, call the `/asset` endpoint with the address of the asset's contract and the token id. The endpoint will return an Asset Object."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/asset/{asset_contract_address}/{token_id}"
    querystring = {}
    if include_orders:
        querystring['include_orders'] = include_orders
    if account_address:
        querystring['account_address'] = account_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieving_events(collection_slug: str=None, auction_type: str=None, asset_contract_address: str='0x4372f4d950d30c6f12c7228ade77d6cc019404c9', token_id: int=309, collection_editor: str=None, occurred_after: int=None, cursor: str=None, account_address: str=None, occurred_before: int=1644800000, only_opensea: bool=None, event_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/events` endpoint provides a list of events that occur on the assets that OpenSea tracks. The "event_type" field indicates what type of event it is (transfer, successful auction, etc)."
    
    """
    url = f"https://opensea-data-query.p.rapidapi.com/api/v1/events"
    querystring = {}
    if collection_slug:
        querystring['collection_slug'] = collection_slug
    if auction_type:
        querystring['auction_type'] = auction_type
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if token_id:
        querystring['token_id'] = token_id
    if collection_editor:
        querystring['collection_editor'] = collection_editor
    if occurred_after:
        querystring['occurred_after'] = occurred_after
    if cursor:
        querystring['cursor'] = cursor
    if account_address:
        querystring['account_address'] = account_address
    if occurred_before:
        querystring['occurred_before'] = occurred_before
    if only_opensea:
        querystring['only_opensea'] = only_opensea
    if event_type:
        querystring['event_type'] = event_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea-data-query.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

