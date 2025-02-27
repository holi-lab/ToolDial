import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_historical_token_balance(walletaddress: str, timestamp: int, chainid: int, offset: int=0, sortby: str='balance', sortbydate: str='desc', tokenaddress: str=None, limit: int=100, sortdirection: str='desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the user's historical portfolio value and balance records along with token info."
    walletaddress: The address that the balance records are tied to. Accepts either address or ENS
        timestamp: Numerical representation of the earliest date the balance records were indexed. Unix timestamp in seconds.
        chainid: Network to filter through balance records.
        offset: Number of records to skip in the query.
        sortby: Specify how to sort assets. E.g. 'balance', 'value'...
        sortbydate: Specify whether to sort records chronologically or reverse-chronologically (asc/desc).
        tokenaddress: The address of a specific token to filter through the balance records.
        limit: The maximum number of balance records to return.
        sortdirection: Specify whether to sort records in ascending or descending order (asc/desc).
        
    """
    url = f"https://uniblock1.p.rapidapi.com/balance/{walletaddress}/historical"
    querystring = {'timestamp': timestamp, 'chainId': chainid, }
    if offset:
        querystring['offset'] = offset
    if sortby:
        querystring['sortBy'] = sortby
    if sortbydate:
        querystring['sortByDate'] = sortbydate
    if tokenaddress:
        querystring['tokenAddress'] = tokenaddress
    if limit:
        querystring['limit'] = limit
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def format_units(value: str, unit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a string representation of a given value formatted with unit digits."
    value: Value to format in string.
        unit: Number of decimals (“18”, ...) or named units (“wei”, “ether”, ...).
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/formatUnits"
    querystring = {'value': value, 'unit': unit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_units(value: str, direction: str, address: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the converted value (decimal units to number or number to decimal units) of a specified amount of a token."
    value: Value to format.
        direction: Specifies conversion method.
    
 * From: Converts a number in units to an actual number.
    
 * To: Converts the actual number to a number in units.
        address: Token contract address that will be used to determine the number of digits to convert the value.
        chainid: Network that the token resides on.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/convertUnits"
    querystring = {'value': value, 'direction': direction, 'address': address, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parse_ether(value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a string representation of a given value, parsed with 18 digits (in terms of wei)."
    value: Value to parse.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/parseEther"
    querystring = {'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_wallet_from_mnemonic(mnemonic: str, index: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a wallet from a mnemonic phrase and index. If no index is provided, it will default to 0."
    mnemonic: Mnemonic phrase to generate a wallet from.
        index: Generate a wallet from the index and the mnemonic phrase pair.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/wallet/fromMnemonic"
    querystring = {'mnemonic': mnemonic, }
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_mnemonic(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a random mnemonic, defaults to 128-bits of entropy"
    
    """
    url = f"https://uniblock1.p.rapidapi.com/wallet/generateMnemonic"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_block_number(chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the current block number."
    chainid: Network to fetch the current block number from.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/block/blockNumber"
    querystring = {'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def look_up_address(chainid: int, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Looks up the ens domain for a crypto address if there is any domain bound with the address."
    chainid: Network to fetch the default ENS domain for.
        address: Address of the user to fetch the default ENS domain for.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/ens/lookupName"
    querystring = {'chainId': chainid, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_balance(contractaddress: str, walletaddress: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the balance of an ERC-20 token."
    contractaddress: Contract address of ERC-20 token.
        walletaddress: The wallet address of the ERC-20 token's owner.
        chainid: Network that the contract address resides on.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/tokenBalance"
    querystring = {'contractAddress': contractaddress, 'walletAddress': walletaddress, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_decimals(address: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the number of decimal places used by an ERC-20 token."
    address: Contract address of ERC-20 token.
        chainid: Network that the contract address resides on.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/decimals"
    querystring = {'address': address, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_native_balance_at_block(blocknumber: int, chainid: int, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the native token balance of an address at a given block."
    blocknumber: Block number to retrieve user balance.
        chainid: Specific network to fetch the user balance for.
        address: The wallet address of the user whose balance is being retrieved.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/nativeBalanceAt"
    querystring = {'blockNumber': blocknumber, 'chainId': chainid, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_total_supply(chainid: int, tokenaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the total supply of an ERC-20 or ERC-721 token."
    chainid: Network that the token resides on.
        tokenaddress: Contract address of the token to retrieve total supply from.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/totalSupply"
    querystring = {'chainId': chainid, 'tokenAddress': tokenaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_approval_for_nft_transfer(owner: str, spender: str, chainid: int, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Determines whether or not a spender address can transfer the NFTs of an owner address."
    owner: The address which we want to determine the approval of all NFTs of.
        spender: The spender of the allowed tokens.
        chainid: Network that the token resides on.
        address: ERC-721/ERC-1155 contract address that will be used to determine whether or not the spender can transfer tokens from the owner.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/approvalForAll"
    querystring = {'owner': owner, 'spender': spender, 'chainId': chainid, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_event_logs(chainid: int, address: str, startblock: int=None, endblock: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the logs of a contract."
    chainid: Network to fetch logs for.
        address: The address to fetch logs from.
        startblock: The block to start fetching logs for.
        endblock: The block to stop fetching logs for.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/log"
    querystring = {'chainId': chainid, 'address': address, }
    if startblock:
        querystring['startBlock'] = startblock
    if endblock:
        querystring['endBlock'] = endblock
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_event_logs_within_block_range(topics: str, chainid: int, startblock: int=None, endblock: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the event logs in a block range, filtered by topics."
    topics: The topic to fetch logs for.
        chainid: Network to fetch logs for.
        startblock: The block to start fetching logs for.
        endblock: The block to stop fetching logs for.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/log/topics"
    querystring = {'topics': topics, 'chainId': chainid, }
    if startblock:
        querystring['startBlock'] = startblock
    if endblock:
        querystring['endBlock'] = endblock
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_collection_nfts(chainid: int, nftaddress: str, offset: int=0, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all NFTs in a collection"
    chainid: Enter the chain ID of the specific NFT to query
        nftaddress: Enter the address of the NFT collection
        offset: Number of records to skip in the query.
        limit: The maximum number of NFT balance records to return.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/nft/inCollection"
    querystring = {'chainId': chainid, 'nftAddress': nftaddress, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_price_history_between_dates(startdate: int, chainid: int, address: str, enddate: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the price chart of a token between two dates in USD."
    startdate: Returns the chart starting at this timestamp. Unix timestamp in seconds.
        chainid: Specific network the desired coin resides on.
        address: The token address of the desired coin.
        enddate: Returns the chart ending at this timestamp. Unix timestamp in seconds.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/price/history/date"
    querystring = {'startDate': startdate, 'chainId': chainid, 'address': address, 'endDate': enddate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transaction_by_hash(hash: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets transaction information of a given hash"
    hash: Transaction hash to get transaction information from.
        chainid: Network to retrieve transaction information from.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/transaction/transactionByHash"
    querystring = {'hash': hash, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transactions_for_user(chainid: int, walletaddress: str, starttimestamp: int=None, interactingwith: str=None, endtimestamp: int=None, offset: int=0, cursor: str=None, sortbydate: str='desc', tokentype: str='any', limit: int=100, direction: str='both', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets transaction records for a specified user."
    chainid: Network to filter through transaction records.
        walletaddress: The user address that the transaction records are tied to. Accepts either address or ENS.
        starttimestamp: Returns transactions starting from this date. Unix timestamp in seconds.
        interactingwith: Specifies the wallet or contract address of the second-party involved in the transaction.
        endtimestamp: Returns transactions before this date. Unix timestamp in seconds.
        offset: Number of records to skip in the query.
        cursor: The cursor returned in the previous response (used to getting the next page).
        sortbydate: Specify whether to sort records chronologically or reverse-chronologically (asc/desc).
        tokentype: Specify token type to filter through transaction records.
        limit: The maximum number of balance records to return.
        direction: Filter transactions based on whether they were sent or received.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/transaction/history/{walletaddress}"
    querystring = {'chainId': chainid, }
    if starttimestamp:
        querystring['startTimestamp'] = starttimestamp
    if interactingwith:
        querystring['interactingWith'] = interactingwith
    if endtimestamp:
        querystring['endTimestamp'] = endtimestamp
    if offset:
        querystring['offset'] = offset
    if cursor:
        querystring['cursor'] = cursor
    if sortbydate:
        querystring['sortByDate'] = sortbydate
    if tokentype:
        querystring['tokenType'] = tokentype
    if limit:
        querystring['limit'] = limit
    if direction:
        querystring['direction'] = direction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_token_balance(chainid: int, walletaddress: str, sortby: str='balance', limit: int=100, cursor: str=None, tokenaddress: str=None, offset: int=0, sortdirection: str='desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the user's current portfolio value and balance records along with token info."
    chainid: Network to filter through balance records.
        walletaddress: The address that the balance records are tied to. Accepts either address or ENS
        sortby: Specify how to sort assets. E.g. 'balance', 'value'...
        limit: The maximum number of balance records to return.
        cursor: The cursor returned in the previous response (used to getting the next page).
        tokenaddress: The address of a specific token to filter through the balance records.
        offset: Number of records to skip in the query.
        sortdirection: Specify whether to sort records in ascending or descending order (asc/desc).
        
    """
    url = f"https://uniblock1.p.rapidapi.com/balance/{walletaddress}"
    querystring = {'chainId': chainid, }
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    if cursor:
        querystring['cursor'] = cursor
    if tokenaddress:
        querystring['tokenAddress'] = tokenaddress
    if offset:
        querystring['offset'] = offset
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ping_server(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check API server status."
    
    """
    url = f"https://uniblock1.p.rapidapi.com/ping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_price(address: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the current price of a token in USD."
    address: The token address of the desired coin.
        chainid: Specific network the desired coin resides on.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/price"
    querystring = {'address': address, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_price_history_by_period(chainid: int, address: str, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the price chart of a token within a given period in USD."
    chainid: Specific network the desired coin resides on.
        address: The token address of the desired coin.
        period: Returns the chart within this time.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/price/history/period"
    querystring = {'chainId': chainid, 'address': address, 'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_wallet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates and gets a new wallet with a random private key, generated from cryptographically secure entropy sources."
    
    """
    url = f"https://uniblock1.p.rapidapi.com/wallet/createWallet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_price_history_between_blocks(chainid: int, endblock: int, startblock: int, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the price chart of a token between two blocks in USD"
    chainid: Specific network the desired coin resides on.
        endblock: Returns the users net worth chart ending at this block.
        startblock: The start block that was requested.
        address: The token address of the desired coin.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/price/history/block"
    querystring = {'chainId': chainid, 'endBlock': endblock, 'startBlock': startblock, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_detailed_block_info(blocknumber: int, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets detailed block information of a given block number"
    blocknumber: Block number to retrieve block.
        chainid: Network to fetch information about a specific block.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/block/blockDetailed"
    querystring = {'blockNumber': blocknumber, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_timestamp_of_block_number(chainid: int, blocknumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a numerical timestamp of a given block number"
    chainid: Network to fetch a timestamp from.
        blocknumber: Block number to retrieve timestamp.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/block/blockTimestamp"
    querystring = {'chainId': chainid, 'blockNumber': blocknumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_number_from_timestamp(chainid: int, timestamp: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the block number closest to the given timestamp."
    chainid: Network to fetch a timestamp from.
        timestamp: Date to retrieve block. Unix timestamp in seconds.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/block/blockNumberFromTimestamp"
    querystring = {'chainId': chainid, 'timestamp': timestamp, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_info_with_transactions(blocknumber: int, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a block with transactions of a given block number"
    blocknumber: Block number to retrieve block.
        chainid: Network to fetch information about a specific block.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/block/blockWithTransactions"
    querystring = {'blockNumber': blocknumber, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_info(chainid: int, blocknumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the block of a given block number"
    chainid: Network to fetch the block from.
        blocknumber: Block number to retrieve timestamp.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/block/block"
    querystring = {'chainId': chainid, 'blockNumber': blocknumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resolve_ens_domain(chainid: int, domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resolves the ens domain to get the address."
    chainid: Network to resolve the domain for.
        domain: ENS domain to resolve.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/ens/resolveName"
    querystring = {'chainId': chainid, 'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_detailed_transaction_by_hash(hash: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets detailed transaction information of a given hash."
    hash: Transaction hash to get transaction information from.
        chainid: Network to retrieve transaction information from.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/transaction/transactionByHashDetailed"
    querystring = {'hash': hash, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_native_balance(chainid: int, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the native token balance of an address."
    chainid: Specific network to fetch the user balance for.
        address: The wallet address of the user whose balance is being retrieved.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/nativeBalance"
    querystring = {'chainId': chainid, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_allowance(spender: str, chainid: int, owner: str, tokenaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the token allowance limit of a spender for tokens owned by another address."
    spender: The spender of the allowance.
        chainid: Network that the token resides on.
        owner: The address which we want to determine the allowance of.
        tokenaddress: The ERC-20 token contract address to check allowance for.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/allowance"
    querystring = {'spender': spender, 'chainId': chainid, 'owner': owner, 'tokenAddress': tokenaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_symbol(address: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the symbol of a given ERC-20 token"
    address: Contract address of ERC-20 token.
        chainid: Network that the contract address resides on.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/token/symbol"
    querystring = {'address': address, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def format_ether(value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a string representation of a given value in terms of ether."
    value: Value to format.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/formatEther"
    querystring = {'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parse_units(unit: str, value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a string representation of value, parsed with unit digits."
    unit: Number of decimals (“18”, ...) or named units (“wei”, “ether”, ...).
        value: Value to parse.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/parseUnits"
    querystring = {'unit': unit, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_code(address: str, chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the code at a given address and network (chainId). This code can be used to distinguish between contract addresses and wallet addresses. Contract addresses return bytecode whereas wallet addresses return "0x"."
    address: Wallet or contract address to retrieve code from.
        chainid: Network that the given address resides on.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/code"
    querystring = {'address': address, 'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commify(value: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a string grouped by three digits separated by comma(s)."
    value: Numerical value to format.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/commify"
    querystring = {'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_gas_price(chainid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the best estimate for gas price to use in a transaction."
    chainid: Network to retrieve gas price estimate for.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/utils/gasPrice"
    querystring = {'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_event_logs_filtered_by_topic(chainid: int, address: str, topics: str, endblock: int=None, startblock: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the event logs by address filtered by topics."
    chainid: Network to fetch logs for.
        address: The address to fetch logs from.
        topics: The topic to fetch logs for.
        endblock: The block to stop fetching logs for.
        startblock: The block to start fetching logs for.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/log/address/topics"
    querystring = {'chainId': chainid, 'address': address, 'topics': topics, }
    if endblock:
        querystring['endBlock'] = endblock
    if startblock:
        querystring['startBlock'] = startblock
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nft_owner(chainid: int, nftaddress: str, nftid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the owner of a specific ERC-721 NFT."
    chainid: Enter the chain ID of the specific NFT to query.
        nftaddress: Enter the address of the NFT contract.
        nftid: Enter the NFT ID of the specific NFT to query.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/nft/nftOwner"
    querystring = {'chainId': chainid, 'nftAddress': nftaddress, 'nftId': nftid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_nft_balance_for_user(chainid: int, walletaddress: str, offset: int=0, limit: int=100, cursor: str=None, nftid: str=None, nftaddress: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the NFT balance of a specific user."
    chainid: Specific network to fetch the user's NFT balance.
        walletaddress: The wallet address of the user whose NFT balances are being retrieved. Accepts either address or ENS.
        offset: Number of records to skip in the query.
        limit: The maximum number of NFT balance records to return.
        cursor: The cursor returned in the previous response (used to getting the next page).
        nftid: Id of the specific NFT to query.
        nftaddress: The address of a specific NFT contract.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/nft/{walletaddress}"
    querystring = {'chainId': chainid, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if cursor:
        querystring['cursor'] = cursor
    if nftid:
        querystring['nftId'] = nftid
    if nftaddress:
        querystring['nftAddress'] = nftaddress
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nft_collection_metadata(chainid: int, collectionaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches NFT collection metadata."
    chainid: Specific network to fetch the user's NFT balance.
        collectionaddress: The address of a specific NFT contract.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/nft/collection/{collectionaddress}/metadata"
    querystring = {'chainId': chainid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_historical_nft_balances(walletaddress: str, chainid: int, timestamp: int, nftid: str=None, nftaddress: str=None, limit: int=100, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the user's historical NFT balance records."
    walletaddress: The wallet address of the user whose NFT balances are being retrieved. Accepts either address or ENS.
        chainid: Specific network to fetch the user's NFT balance.
        timestamp: Date to retrieve user's NFT balances. Unix timestamp in seconds.
        nftid: Id of the specific NFT to query.
        nftaddress: The address of a particular NFT contract.
        limit: The maximum number of NFT balance records to return.
        offset: Number of rows to skip in query.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/nft/{walletaddress}/historical"
    querystring = {'chainId': chainid, 'timestamp': timestamp, }
    if nftid:
        querystring['nftId'] = nftid
    if nftaddress:
        querystring['nftAddress'] = nftaddress
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nft_info(chainid: int, nftaddress: str, nftid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets information of a specific NFT."
    chainid: Enter the chain ID of the specific NFT to query.
        nftaddress: Enter the address of the NFT contract.
        nftid: Enter the NFT ID of the specific NFT to query.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/nft/nftInfo"
    querystring = {'chainId': chainid, 'nftAddress': nftaddress, 'nftId': nftid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nft_transfers_for_user(chainid: int, walletaddress: str, nftaddress: str=None, direction: str='both', sortbydate: str='desc', limit: int=100, offset: int=0, cursor: str=None, nftid: str=None, interactingwith: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets NFT transfer records for a specified user."
    chainid: Network to filter through NFT transfer records.
        walletaddress: The user address that the transaction records are tied to. Accepts either address or ENS.
        nftaddress: Specify NFT address to filter for in the NFT transfer records.
        direction: Filter transactions based on whether they were sent or received.
        sortbydate: Specify whether to sort NFT transfers chronologically or reverse-chronologically (asc/desc).
        limit: The maximum number of NFT transfer records to return.
        offset: Number of records to skip in the query.
        cursor: The cursor returned in the previous response (used to getting the next page).
        nftid: Specify NFT ID to filter for in the NFT transfer records.
        interactingwith: Specifies the wallet or contract address of the second-party involved in the NFT transfer.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/transfer/{walletaddress}/nft"
    querystring = {'chainId': chainid, }
    if nftaddress:
        querystring['nftAddress'] = nftaddress
    if direction:
        querystring['direction'] = direction
    if sortbydate:
        querystring['sortByDate'] = sortbydate
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if cursor:
        querystring['cursor'] = cursor
    if nftid:
        querystring['nftId'] = nftid
    if interactingwith:
        querystring['interactingWith'] = interactingwith
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_collections(chainid: int, walletaddress: str, nftaddress: str=None, limit: int=100, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all NFT collections of a specific user."
    chainid: Enter the chain ID of the specific NFT to query.
        walletaddress: The wallet address of the user whose NFT balances are being retrieved. Accepts either address or ENS.
        nftaddress: Enter the address of the NFT collection.
        limit: The maximum number of NFT balance records to return.
        offset: Number of records to skip in the query.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/nft/{walletaddress}/collections"
    querystring = {'chainId': chainid, }
    if nftaddress:
        querystring['nftAddress'] = nftaddress
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_transfers_for_user(chainid: int, walletaddress: str, direction: str='both', cursor: str=None, interactingwith: str=None, offset: int=0, sortbydate: str='desc', tokenaddress: str=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets token transfer records for a specified user."
    chainid: Network to filter through token transfer records.
        walletaddress: The user address that the transaction records are tied to. Accepts either address or ENS.
        direction: Filter transactions based on whether they were sent or received.
        cursor: The cursor returned in the previous response (used to getting the next page).
        interactingwith: Specifies the wallet or contract address of the second-party involved in the token transfer.
        offset: Number of records to skip in the query.
        sortbydate: Specify whether to sort token transfers chronologically or reverse-chronologically (asc/desc).
        tokenaddress: Specify token address to filter for in the token transfer records.
        limit: The maximum number of token transfer records to return.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/transfer/{walletaddress}/tokens"
    querystring = {'chainId': chainid, }
    if direction:
        querystring['direction'] = direction
    if cursor:
        querystring['cursor'] = cursor
    if interactingwith:
        querystring['interactingWith'] = interactingwith
    if offset:
        querystring['offset'] = offset
    if sortbydate:
        querystring['sortByDate'] = sortbydate
    if tokenaddress:
        querystring['tokenAddress'] = tokenaddress
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nft_transfers_for_nft_collection(nftaddress: str, chainid: int, limit: int=100, offset: int=0, cursor: str=None, sortbydate: str='desc', nftid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets NFT transfer records for a specified NFT collection inclusive of token ID and addresses involved."
    nftaddress: The contract address that the transaction records are tied to.
        chainid: Network to filter through NFT transfer records.
        limit: The maximum number of NFT transfer records to return.
        offset: Number of records to skip in the query.
        cursor: The cursor returned in the previous response (used to getting the next page).
        sortbydate: Specify whether to sort NFT transfers chronologically or reverse-chronologically (asc/desc).
        nftid: Specify NFT ID to filter for in the NFT transfer records.
        
    """
    url = f"https://uniblock1.p.rapidapi.com/transfer/{nftaddress}"
    querystring = {'chainId': chainid, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if cursor:
        querystring['cursor'] = cursor
    if sortbydate:
        querystring['sortByDate'] = sortbydate
    if nftid:
        querystring['nftId'] = nftid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uniblock1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

