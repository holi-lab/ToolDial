import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getassetsbycontractinformation(token_id: str=None, contract_address: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request will return asset_id wrapped with return code and error code"
    token_id: TokenID of NFT(ERC721/ERC721M need this field)
        contract_address: Contract address of token (when type is ERC20/ERC721/ERC721M need this field, if type is ETH, then it can be ignore)
        type: Type of token, Possible enum values is ETH, ERC20, ERC721, ERC721M
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/assetid"
    querystring = {}
    if token_id:
        querystring['token_id'] = token_id
    if contract_address:
        querystring['contract_address'] = contract_address
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassetsdetailbyassetid(asset_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API can query every asset_id that exists on reddio's system, if the FT/NFT is not on reddio's system, then the API will not work well
		
		The API returns assets detail including contract_address, type, token_id(if type is ERC721 or ERC721M), quantum and token owners(owns by who)"
    asset_id: The asset id you want to retrieve information
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/assetid/{asset_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvaultsdetailbyvaultid(vault_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/vaults/{vault_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvaultsbystark_keyandasset_id(stark_keys: str=None, asset_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    stark_keys: stark_keys can be multi stark_key, separated by ","
        asset_id: the asset id
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/vaults"
    querystring = {}
    if stark_keys:
        querystring['stark_keys'] = stark_keys
    if asset_id:
        querystring['asset_id'] = asset_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorderbyorderid(order_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API will return the order related to the order id
		
		There are many informations it returns
		
		| **field** | **description** |
		| --- | --- |
		| order_id | the id of the order, it is the same value as sequence id |
		| stark_key | the stark_key of who place the order |
		| price | the price of the NFT, when need a human-readable value, display_price is the human-readable value |
		| display_price | the price that human-readable |
		| direction | 0 for sell, 1 for buy |
		| amount | the amount to buy or sell |
		| un_filled | how many tokens are unfilled |
		| symbol | a structure contains base and quota token, also have token id in it |
		| fee_rate | the fee_rate |
		| token_type | the token type |
		| token_id | the token ID |
		| order_state | the state of the order |
		| resp | if there has some error, the field will be set |
		
		The enum value of order_state is
		
		| Status | Value |
		| --- | --- |
		| Placed | 0 |
		| Canceled | 1 |
		| Filled | 2 |
		| PartiallyFilled | 3 |
		| ConditionallyCanceled | 4 |"
    order_id: the order id you want to query
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/order"
    querystring = {}
    if order_id:
        querystring['order_id'] = order_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listorders(contract_address: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    contract_address: the contract address of order you want to list
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/orders"
    querystring = {}
    if contract_address:
        querystring['contract_address'] = contract_address
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwallets(count: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The return field as following
		
		| **field** | **description** |
		| --- | --- |
		| stark_key | the stark public key |
		| stark_private_key | the stark private key |"
    count: how many wallets you want to generated
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/wallets"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecordbysignature(s: str=None, r: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    s: signature S
        r: signature R
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/record/by/signature"
    querystring = {}
    if s:
        querystring['s'] = s
    if r:
        querystring['r'] = r
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecordbysequenceid(sequence_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    sequence_id: The sequence id you want to query
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/record"
    querystring = {}
    if sequence_id:
        querystring['sequence_id'] = sequence_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecordsbystark_key(stark_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    stark_key: Which stark_key's record you want to query
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/records"
    querystring = {}
    if stark_key:
        querystring['stark_key'] = stark_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aggregationbalanceofastark_keybycontractaddress(stark_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API aggregation balances by contract address, for previous API, each NFT have an asset_id, but in this API. NFT do not have asset_id when return, but has token id"
    stark_key: The stark_key you want to check balance
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v2/balances"
    querystring = {}
    if stark_key:
        querystring['stark_key'] = stark_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallcollections(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API lists all collections on Reddio's layer2 system.
		
		The return field is like the following descrption
		
		| **field** | **description** |
		| --- | --- |
		| contract_address | The contract address |
		| symbol | The symbol of |
		| type | ERC721 or ERC721M |
		| total_supply | Total Supply of the token |
		| base_uri | The base uri of NFT |
		| asset_type | The asset type is calculated by reddio, the definition is [here](https://docs.starkware.co/starkex/spot/shared/starkex-specific-concepts.html#computing_asset_info_asset_type_asset_id) |"
    
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/collections"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnoncebystark_key(stark_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    stark_key: which stark_key's nonce you want to get
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/nonce"
    querystring = {}
    if stark_key:
        querystring['stark_key'] = stark_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balancesofstark_key(limit: str=None, stark_key: str=None, contract_address: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API retrieves the balance of a stark_key, each return value are as the previous field"
    limit: set limit per page
        stark_key: The stark key
        contract_address: filter balance by contract address
        page: which page(starts by 1)
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/balances"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if stark_key:
        querystring['stark_key'] = stark_key
    if contract_address:
        querystring['contract_address'] = contract_address
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarketplaces(project_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/project/{project_uuid}/marketplace"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balanceofstark_keyandasset_id(stark_key: str=None, asset_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API retrieves the balance of a stark_key and asset_id
		
		The return values description
		
		| **key** | **description** |
		| --- | --- |
		| asset_id | The asset's id |
		| contract_address | The contract address of ERC20/ERC721/ERC721, if the asset_id's type is ETH, it will return "eth" as a placeholder |
		| token_id | The NFT token ID(this field is set when type is ERC721 or ERC721M) |
		| base_uri | The NFT base uri(this field is set when type is ERC721 or ERC721M) |
		| available | The available balance on layer 2. |
		| frozen | The frozen balance on layer 2. |
		| type | Type of token, Possible enum values is ETH, ERC20, ERC721, ERC721M |
		| symbol | The symbol of ERC721/ERC721M or ERC20("ETH" for ETH) |
		| decimals | The decimals of asset, for ETH is 18, ERC721/ERC721M is 1, and for ERC20, it is based the contracts, [here](https://medium.com/@jgm.orinoco/understanding-erc-20-token-contracts-a809a7310aa5) is more explain about decimals on Ethernum |
		| quantum | quantum is a starkex concept, it is auto-set when the contract is registered to reddio. [here](https://docs.starkware.co/starkex/spot/shared/starkex-specific-concepts.html) is more explain about quantum |
		| display_value | deprecated,use `available` instead |
		| display_frozen | deprecated,use `frozen` instead |
		| balance_available | The available balance represents on layer 2. Keep attention that it need to be converted with quantum and decimals if you need to use this field. All SDK hands this value, so most of the time you do not need to care about it. |
		| balance_frozen | The frozen balance represents on layer 2. Keep attention that it need to be converted with quantum and decimals if you need to use this field. All SDK hands this value, so most of the time you do not need to care about it. |"
    stark_key: The stark_key which you want to check balance
        asset_id: Which asset you want to check, you can get asset id in prevous assets APIs
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/balance"
    querystring = {}
    if stark_key:
        querystring['stark_key'] = stark_key
    if asset_id:
        querystring['asset_id'] = asset_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorderinfo(stark_key: str=None, contract2: str=None, contract1: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is to get order related information, such as the asset id of base/quota tokens. nonce and vaults related to stark_key
		
		**If you use SDKs(including java/unity/js), you do not need to care about the order info interface, it is already handled by SDK internally**
		
		the contract1 and contract2 values are like
		
		{type}:{contract_address}:{token_id}
		
		There are many informations it returns
		
		| **field** | **description** |
		| --- | --- |
		| fee_rate | the upper fee_rate allowed by reddio and marketplace. that is, if reddio & marketplace charge's more then the rate, the order will fail |
		| base_token | base token asset id, in general, if you trade using ETH for NFT, then the base token is ETH's asset id |
		| fee_token | which token as a fee, in general, it is the same as base_token |
		| lower_limit | the lower price of the NFT |
		| nonce | the nonce of the stark_key |
		| contracts | the contracts info of contract1 and contract2 |
		| asset_ids | the asset ids of contract1 and contract2 |
		| vault_ids | first vault id is generated by the stark_key and first asset id, the second vault id is generated by the stark_key and second asset id(you can see the asset_ids above) |"
    stark_key: the stark_key that want to buy or sell NFT
        contract2: the quota contract and token id, the vaule can like ERC721:0xA21B04B6dbd1174155E242434B3Df2EeD74BaFb2:28
        contract1: the base contract info, if sell using ETH, then the vaule should be ETH:eth:1
        
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/order/info"
    querystring = {}
    if stark_key:
        querystring['stark_key'] = stark_key
    if contract2:
        querystring['contract2'] = contract2
    if contract1:
        querystring['contract1'] = contract1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcolloction_snftowners(contract_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The api returns a collection's nft owners.
		
		The response field description
		
		| **field** | **description** |
		| --- | --- |
		| contract_address | the contract_address of the collection |
		| token_id | the NFT token id |
		| owner | the owner of the NFT |
		| symbol | the NFT's symbol |
		| asset_id | the token's asset id |"
    
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/nfts/{contract_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listfilesunderproject(project_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will list files under the project, the response item like
		
		| **field** | **description** |
		| --- | --- |
		| cid | [Content Identifiers](https://docs.ipfs.tech/concepts/content-addressing/#what-is-a-cid) |
		| filename | The stored file name |
		| created_at | first create time |
		| filesize | file size(in byte) |
		| endpoints | the endpoints you can view the file |"
    
    """
    url = f"https://reddio-nft-token-and-ipfs.p.rapidapi.com/v1/project/{project_uuid}/storage"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddio-nft-token-and-ipfs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

