import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_active_loans_offers(collection: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve the active loan offer levels for a collection.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/blend/active-liens/{collection}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_blur_events(filters: str='{"count":50,"tokenId":"541","contractAddress":"0x60e4d786628fea6478f785a6d7e704777c86a7c6","eventFilter":{"mint":{},"sale":{},"transfer":{},"orderCreated":{}}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve events, customisable.
		
		**Filter examples:**
		{"count":50,"eventFilter":{"mint":{},"sale":{},"transfer":{},"orderCreated":{}}}
		
		{"count":50,"tokenId":"541","contractAddress":"0x60e4d786628fea6478f785a6d7e704777c86a7c6","eventFilter":{"mint":{},"sale":{},"transfer":{},"orderCreated":{}}}
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/activity/event-filter"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Blur Search Query
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_aggregated_loan_offers(collection: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve the aggregated loan offer levels for a collection.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/blend/aggregated-loan-offers/{collection}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_blur_collections(filters: str='{"sort":"VOLUME_ONE_DAY","order":"DESC"}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve information about all collection, including real time statistics such as floor price.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/collections/"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_user_collection_bids(walletaddress: str, filters: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve the active collection bids to a user wallet address.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/collection-bids/user/{walletaddress}"
    querystring = {'filters': filters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_bids(collection: str, filters: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve the collection bid price levels for a collection.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/collections/{collection}/executable-bids"
    querystring = {'filters': filters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection(collection: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve more in-depth information about an individual collection, including real time statistics such as floor price.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    collection: Enter contract address or slug
        
    """
    url = f"https://opensea15.p.rapidapi.com/v1/collections/{collection}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_projects(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve projects"
    
    """
    url = f"https://opensea15.p.rapidapi.com/projects"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_orderbook_depth(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve orderbook depth"
    
    """
    url = f"https://opensea15.p.rapidapi.com/collections/{slug}/orderbook/depth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x2y2_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "x2y2 API endpoint (https://api.x2y2.io)
		Make over 1000 requests / minute
		
		Simply copy the path into the x2y2_path header.
		
		![](https://i.ibb.co/WKYfTDy/x2y2-path-get-header.png)"
    
    """
    url = f"https://opensea15.p.rapidapi.com/uniapi/xy3/v1/offer/list?contractAddress=0x34d85c9CDeB23FA97cb08333b511ac86E1C4E258&tokenId=28807&duration=0&page=1&pageSize=10&sort=desc&order=createdAt&isSufficient=1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_offers(slug: str, next: str=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all open offers for a collection
		
		This endpoint is used to get all open offers for a given collection. It returns: all offers on individual NFTs, all collection offers, and all trait offers. The offers are sorted by ascending date (oldest listings first) -- at this time, we do not support any other sorting methods.
		
		The response has two fields: offers and an optional next cursor in case the number of offers is greater than our default page size (100). This endpoint returns only a subset of the fields found in our orders endpoint. Here is the model for each offer:
		
		order_hash
		chain
		criteria: specifies if the offer was for an individual NFT, a collection offer, or a trait offer
		protocol_data"
    
    """
    url = f"https://opensea15.p.rapidapi.com/v2/offers/collection/{slug}/all"
    querystring = {}
    if next:
        querystring['next'] = next
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all_listings(slug: str, limit: int=20, next: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active listings for a collection
		
		This endpoint is used to get all active listings for a given collection. The listings are sorted by ascending date (oldest listings first) -- at this time, we do not support any other sorting methods.
		
		The response has two fields: listings and an optional next cursor in case the number of active listings is greater than our default page size (100). This endpoint returns only a subset of the fields found in our orders endpoint. Here is the model for each listing
		
		order_hash
		chain
		type: can be basic, english, or dutch
		price
		protocol_data
		Sample URL
		https://api.opensea.io/v2/listings/collection/cool-cats-nft/all"
    
    """
    url = f"https://opensea15.p.rapidapi.com/v2/listings/collection/{slug}/all"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_trait_offers(slug: str, value: str='classy_1', type: str='tier', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get all trait offers for a specified collection.
		
		What are trait offers?
		A trait offer is an offer that applies to every item in a collection with the specified trait. For example, this is a filter that shows every Cool Cat NFT where the type is 'tier' and the value is 'classy_1'.
		
		Below is a response when querying trait offers for Cool Cats where trait type is 'tier' and the value is 'classy_1'.
		
		Note: this example query might not return any data if there are no open offers for that trait.
		
		Sample URL
		https://api.opensea.io/v2/offers/collection/cool-cats-nft/traits?type=tier&value=classy_1"
    
    """
    url = f"https://opensea15.p.rapidapi.com/v2/offers/collection/{slug}/traits"
    querystring = {}
    if value:
        querystring['value'] = value
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_offers(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get collection offers.
		
		What are collection offers?
		If you'd like to purchase into a collection but don't have a particular NFT in mind, collection offers allow you to make an offer that applies to all NFTs in a collection.
		
		Below is a response when querying collection offers for Cool Cats."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v2/offers/collection/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_listings(chain: str, maker: str=None, order_direction: str='desc', order_by: str='created_date', listed_before: str=None, listed_after: str=None, taker: str=None, limit: str=None, token_ids: str=None, asset_contract_address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch the set of active listings on a given NFT for the Seaport contract."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v2/orders/{chain}/seaport/listings"
    querystring = {}
    if maker:
        querystring['maker'] = maker
    if order_direction:
        querystring['order_direction'] = order_direction
    if order_by:
        querystring['order_by'] = order_by
    if listed_before:
        querystring['listed_before'] = listed_before
    if listed_after:
        querystring['listed_after'] = listed_after
    if taker:
        querystring['taker'] = taker
    if limit:
        querystring['limit'] = limit
    if token_ids:
        querystring['token_ids'] = token_ids
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_offers(chain: str, asset_contract_address: str='0xbce3781ae7ca1a5e050bd9c4c77369867ebc307e', limit: str=None, token_ids: str='1475', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch the set of active offers on a given NFT for the Seaport contract."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v2/orders/{chain}/seaport/offers"
    querystring = {}
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if limit:
        querystring['limit'] = limit
    if token_ids:
        querystring['token_ids'] = token_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_assets(limit: int=20, token_ids: str=None, asset_contract_addresses: str=None, collection_editor: str=None, include_orders: bool=None, asset_contract_address: str=None, order_direction: str='desc', collection_slug: str=None, collection: str='ongakucraft', cursor: str=None, owner: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a set of NFTs based on the specified filter parameters.
		
		Auctions created on OpenSea don't use an escrow contract, which enables gas-free auctions and allows users to retain ownership of their items while they're on sale. In these cases, some NFTs from opensea.io may not appear in the API until a transaction has been completed."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/assets"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if token_ids:
        querystring['token_ids'] = token_ids
    if asset_contract_addresses:
        querystring['asset_contract_addresses'] = asset_contract_addresses
    if collection_editor:
        querystring['collection_editor'] = collection_editor
    if include_orders:
        querystring['include_orders'] = include_orders
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if order_direction:
        querystring['order_direction'] = order_direction
    if collection_slug:
        querystring['collection_slug'] = collection_slug
    if collection:
        querystring['collection'] = collection
    if cursor:
        querystring['cursor'] = cursor
    if owner:
        querystring['owner'] = owner
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collections(asset_owner: str='0x2bf699087a0d1d67519ba86f960fecd80d59c4d7', offset: int=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a list of all the collections supported and vetted by OpenSea. To include all collections relevant to a user (including non-whitelisted ones), set the owner param.
		
		Each collection in the response has an attribute called primary_asset_contracts with info about the smart contracts belonging to that collection. For example, ERC-1155 contracts maybe have multiple collections all referencing the same contract, but many ERC-721 contracts may all belong to the same collection (dapp).
		
		You can also use this endpoint to find which dapps an account uses, and how many items they own in each in a single API call."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/collections"
    querystring = {}
    if asset_owner:
        querystring['asset_owner'] = asset_owner
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_an_asset(asset_contract_address: str, token_id: str, include_orders: bool=None, account_address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch information about a single NFT, based on its contract address and token ID. The response will contain an Asset Object."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/asset/{asset_contract_address}/{token_id}/"
    querystring = {}
    if include_orders:
        querystring['include_orders'] = include_orders
    if account_address:
        querystring['account_address'] = account_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_fees(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the various collection fees
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/collections/{slug}/fees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eth_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current ETH price, can beused to generate a new __cf_bm token to be passed in every request as a header called '__cf_bm'
		
		https://developers.cloudflare.com/fundamentals/get-started/reference/cloudflare-cookies/"
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/prices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_rewards(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve if bid rewards
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/collections/{slug}/rewards"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_fees(feerequests: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the various contract fees
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/fees"
    querystring = {'feeRequests': feerequests, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_airdrop_leaderboard(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the current airdrop leaderboard list
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/rewards/leaderboard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_traits(contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve trait information about a single collection.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/traits/{contract_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_activity(filters: str='{"count":25,"contractAddress":"0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d","eventTypes":["SALE","ORDER_CREATED"]}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve activity for an individual collection.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/activity"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_tokens_listings(collection: str, filters: str='{"cursor":null,"traits":[],"hasAsks":true}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a set of NFTs based on the specified filter parameters.
		
		Filter for Listed Tokens: {"cursor":null,"traits":[],"hasAsks":true}
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    collection: Enter contract address or slug
        
    """
    url = f"https://opensea15.p.rapidapi.com/v1/collections/{collection}/tokens"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retreive_portfolio(filters: str, walletaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve portfolio details for a single wallet address.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/portfolio/{walletaddress}/owned"
    querystring = {'filters': filters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_user_airdrop_rewards(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve user airdrop rewards
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/user/rewards/wallet-compact"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_stats(collection_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be used to fetch stats for a specific collection, including real-time floor price data."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/collection/{collection_slug}/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_owners(asset_contract_address: str, token_id: str, limit: str='20', cursor: str=None, order_by: str='created_date', order_direction: str='desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to obtain the entire list of owners for an NFT. Results will also include the quantity owned."
    
    """
    url = f"https://opensea15.p.rapidapi.com/v1/asset/{asset_contract_address}/{token_id}/owners"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if cursor:
        querystring['cursor'] = cursor
    if order_by:
        querystring['order_by'] = order_by
    if order_direction:
        querystring['order_direction'] = order_direction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_events(account_address: str=None, occurred_after: str=None, auction_type: str=None, event_type: str=None, collection_slug: str=None, only_opensea: bool=None, collection_editor: str=None, token_id: int=863, occurred_before: int=None, cursor: str=None, asset_contract_address: str='0xed5af388653567af2f388e6224dc7c4b3241c544', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a list of events that occur on the NFTs that are tracked by OpenSea. The event_type field indicates the type of event (transfer, successful auction, etc) and the results are sorted by event timestamp.
		
		Note that due to block reorganizations, recent events (less than 10 minutes old) may not reflect the final state of the blockchain."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/events"
    querystring = {}
    if account_address:
        querystring['account_address'] = account_address
    if occurred_after:
        querystring['occurred_after'] = occurred_after
    if auction_type:
        querystring['auction_type'] = auction_type
    if event_type:
        querystring['event_type'] = event_type
    if collection_slug:
        querystring['collection_slug'] = collection_slug
    if only_opensea:
        querystring['only_opensea'] = only_opensea
    if collection_editor:
        querystring['collection_editor'] = collection_editor
    if token_id:
        querystring['token_id'] = token_id
    if occurred_before:
        querystring['occurred_before'] = occurred_before
    if cursor:
        querystring['cursor'] = cursor
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rarity_tools(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rarity Tools Collection Items Endpoint , to  use you need to understand the source code of the webpage and reconstruct."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v0/collections/{slug}/artifacts/data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_information(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape all the HTML information from the NFT asset page without having to use chrome , scrape 10,000  NFTs images, & metadata in under 11min (*using top plan)"
    
    """
    url = f"https://opensea15.p.rapidapi.com/{user}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def collection_information(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape all the HTML information from the NFT collection page without having to use chrome."
    
    """
    url = f"https://opensea15.p.rapidapi.com/collection/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asset_information(contract_address: str, token_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape all the HTML information from the NFT asset page without having to use chrome , scrape 10,000  NFTs images, & metadata in under 11min (*using top plan)"
    
    """
    url = f"https://opensea15.p.rapidapi.com/assets/ethereum/{contract_address}/{token_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_bundles(on_sale: bool=None, limit: str='1', offset: str=None, asset_contract_address: str=None, owner: str=None, asset_contract_addresses: str=None, token_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bundles are groups of NFTs for sale on OpenSea. You can buy them all at once in one transaction, and you can create them without any transactions or gas, as long as you've already approved the NFTs inside."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/bundles"
    querystring = {}
    if on_sale:
        querystring['on_sale'] = on_sale
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if asset_contract_address:
        querystring['asset_contract_address'] = asset_contract_address
    if owner:
        querystring['owner'] = owner
    if asset_contract_addresses:
        querystring['asset_contract_addresses'] = asset_contract_addresses
    if token_ids:
        querystring['token_ids'] = token_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_collection(collection_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve more in-depth information about an individual collection, including real time statistics such as floor price."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/collection/{collection_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_contract(asset_contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to fetch detailed information about a specified contract."
    
    """
    url = f"https://opensea15.p.rapidapi.com/api/v1/asset_contract/{asset_contract_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opensea15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

