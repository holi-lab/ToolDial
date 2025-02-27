import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_collection_activity(filters: str='{"count":25,"contractAddress":"0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d","eventTypes":["SALE","ORDER_CREATED"]}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve activity for an individual collection.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/activity"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/traits/{contract_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retreive_portfolio_collections(walletaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve collections owned for a single wallet address.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/portfolio/{walletaddress}/collections"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/activity/event-filter"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retreive_portfolio(walletaddress: str, filters: str='{}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve portfolio details for a single wallet address.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/portfolio/{walletaddress}/owned"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_a_collection(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve more in-depth information about an individual collection, including real time statistics such as floor price.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/collections/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_tokens(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a set of NFTs based on the specified filter parameters.
		
		Filter for Listed Tokens: {"cursor":null,"traits":[],"hasAsks":true}
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/collections/{slug}/tokens"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_fees(feerequests: str='[{"tokenId":null,"contractAddress":"0x5d62fb8dcd9b480960f55956fbdd8d9f07f2b402"}]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the various contract fees
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/fees"
    querystring = {}
    if feerequests:
        querystring['feeRequests'] = feerequests
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/rewards/leaderboard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/collections/{slug}/fees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_collection_pricing_chart(spanms: str='86400000', collectionid: str='0xed5af388653567af2f388e6224dc7c4b3241c544', intervalms: str='300000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve collection pricing chart data
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/charts/everything"
    querystring = {}
    if spanms:
        querystring['spanMs'] = spanms
    if collectionid:
        querystring['collectionId'] = collectionid
    if intervalms:
        querystring['intervalMs'] = intervalms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_asset(contractaddress: str, tokenid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve information about a single NFT.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/collections/{contractaddress}/tokens/{tokenid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retreive_collections(filters: str='{"sort":"VOLUME_ONE_DAY","order":"DESC"}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve information about all collection, including real time statistics such as floor price.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/collections/"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retreive_collection_prices(collection: str, filters: str='filters: {"traits":[],"hasAsks":true}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve collection prices
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/collections/{collection}/prices"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_active_user_lend_eth_offers(collection: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve active user lend ETH offers
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/portfolio/{collection}/loan-offers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_active_loans_offers(collection: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve the active loan offer levels for a collection.
		
		To be able to use the blur endpoints you need an authToken.
		You can generate one by using the 'Retrieve authchallenge' endpoint with your wallet address in the body of the request.
		Once you get the response, sign the 'message' string with ethers.js
		const signature = await ethersSigner.signMessage(response.data.message);
		Then use the 'Retrieve authtoken' endpoint to get your authToken, by inputting the response parameters from 'Retrieve authchallenge' plus the generated signature."
    
    """
    url = f"https://blur.p.rapidapi.com/v1/blend/active-liens/{collection}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/blend/aggregated-loan-offers/{collection}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/collections/{slug}/rewards"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/user/rewards/wallet-compact"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    
    """
    url = f"https://blur.p.rapidapi.com/v1/collections/{collection}/tokens"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
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
    url = f"https://blur.p.rapidapi.com/v1/collection-bids/user/{walletaddress}"
    querystring = {'filters': filters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retreive_collection_bids(slug: str, filters: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive collection bids"
    
    """
    url = f"https://blur.p.rapidapi.com/v1/collections/{slug}/executable-bids"
    querystring = {'filters': filters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

