import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def estimate_transaction_fee_v2(confirmationtarget: int, blockchain: str, conservative: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an estimated transaction fee for a specific confirmation target.
		If you want your transaction to be included in the next block, then you give 1 as parameter. If it is not urgent, then you can wait a bit longer and get an estimation for the fifth next block."
    confirmationtarget: Number of blocks in which the transaction should be confirmed
        blockchain: Blockchain name
        conservative: Sets fee estimation mode for Bitcoin-like coins. If set to false, fee estimate mode is ECONOMICAL, true means CONSERVATIVE mode which is the default. Has no effect on Ethereum-like coins More info: https://bitcoincore.org/en/doc/24.0.0/rpc/util/estimatesmartfee/
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/estimatefee/{confirmationtarget}"
    querystring = {}
    if conservative:
        querystring['conservative'] = conservative
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nft_metadata_v2(blockchain: str, nftcontract: str, nfttokenid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Only works on Ethereum-like blockchains (currently ethereum and bsc)
		
		Get metadata like name or description for a specified contract and token ID.
		The resulting data contains a link which can then be used to request the IPFS link for the actual image to display in a block explorer for example.
		
		Note: this route was implemented by us and is therefore not yet supported by existing blockbook clients."
    blockchain: NFT-compatible blockchain name
        nftcontract: Address of NFT contract
        nfttokenid: Unique token ID of NFT
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/nft/{nftcontract}/{nfttokenid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_raw_block_data_v2(blockhashorheight: str, blockchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the raw hex-encoded block data for a given block hash or height"
    blockhashorheight: Block hash or height
        blockchain: Blockchain name
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/rawblock/{blockhashorheight}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_utxo_v2(addressorxpub: str, blockchain: str, confirmed: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns array of unspent transaction outputs of address or xpub, applicable only for Bitcoin-type coins. By default, the list contains both confirmed and unconfirmed transactions. The query parameter confirmed=true disables return of unconfirmed transactions. The returned utxos are sorted by block height, newest blocks first. For xpubs or output descriptors, the response also contains address and derivation path of the utxo.
		
		
		
		Unconfirmed utxos do not have field height, the field confirmations has value 0 and may contain field lockTime, if not zero.
		
		Coinbase utxos have field coinbase set to true, however due to performance reasons only up to minimum coinbase confirmations limit (100). After this limit, utxos are not detected as coinbase."
    addressorxpub: Address or XPUB
        blockchain: Blockchain name
        confirmed: confirmed=true disables return of unconfirmed transactions
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/utxo/{addressorxpub}"
    querystring = {}
    if confirmed:
        querystring['confirmed'] = confirmed
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_address_v2(address: str, blockchain: str, details: str='txids', page: int=1, fromblock: int=10, secondary: str='usd', toblock: int=100, pagesize: int=1000, contract: str='0xdAC17F958D2ee523a2206206994597C13D831ec7', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns balances and transactions of an address. The returned transactions are sorted by block height, newest blocks first.
		
		The **details** query parameter can specify the level of details returned by the request (default: "txids").
		Possible values are:
		
		**basic**: return only xpub balances, without any derived addresses and transactions
		
		**tokens**: basic + tokens (addresses) derived from the xpub, subject to tokens parameter
		
		**tokenBalances**: basic + tokens (addresses) derived from the xpub with balances, subject to tokens parameter
		
		**txids**: tokenBalances + list of txids, subject to from, to filter and paging
		
		**txs**: tokenBalances + list of transaction with details, subject to from, to filter and paging
		"
    address: Wallet address
        blockchain: Blockchain name
        details: specifies level of details returned by request
        page: specifies page of returned transactions, starting from 1. If out of range, Blockbook returns the closest possible page.
        fromblock: filter of the returned transactions from block height to block height (default no filter)
        secondary: specifies secondary (fiat) currency in which the token and total balances are returned in addition to crypto values
        toblock: filter of the returned transactions from block height to block height (default no filter)
        pagesize: number of transactions returned by call (default and maximum 1000)
        contract: return only transactions which affect specified contract (applicable only to coins which support contracts)
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/address/{address}"
    querystring = {}
    if details:
        querystring['details'] = details
    if page:
        querystring['page'] = page
    if fromblock:
        querystring['fromBlock'] = fromblock
    if secondary:
        querystring['secondary'] = secondary
    if toblock:
        querystring['toBlock'] = toblock
    if pagesize:
        querystring['pageSize'] = pagesize
    if contract:
        querystring['contract'] = contract
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_xpub_v2(blockchain: str, xpub: str, pagesize: int=1000, secondary: str='usd', toblock: int=100, fromblock: int=10, tokens: str='nonzero', details: str='txids', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns balances and transactions of an xpub or output descriptor, applicable only for Bitcoin-type coins.
		
		Blockbook supports BIP44, BIP49, BIP84 and BIP86 (Taproot) derivation schemes, using either xpubs or output descriptors (see https://github.com/bitcoin/bitcoin/blob/master/doc/descriptors.md)
		
		Note: usedTokens always returns total number of used addresses of xpub.
		
		Detailed documentation found here: https://github.com/trezor/blockbook/blob/master/docs/api.md#get-xpub"
    blockchain: Blockchain name
        xpub: xpub or output descriptor, applicable only for Bitcoin-type coins
        pagesize: number of transactions returned by call (default and maximum 1000)
        secondary: specifies secondary (fiat) currency in which the token and total balances are returned in addition to crypto values
        toblock: filter of the returned transactions from block height to block height (default no filter)
        fromblock: filter of the returned transactions from block height to block height (default no filter)
        tokens: specifies what tokens (xpub addresses) are returned by the request (default nonzero)
        details: specifies level of details returned by request
        page: specifies page of returned transactions, starting from 1. If out of range, Blockbook returns the closest possible page.
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/xpub/{xpub}"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if secondary:
        querystring['secondary'] = secondary
    if toblock:
        querystring['toBlock'] = toblock
    if fromblock:
        querystring['fromBlock'] = fromblock
    if tokens:
        querystring['tokens'] = tokens
    if details:
        querystring['details'] = details
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_balance_history_v2(addressorxpub: str, blockchain: str, groupby: int=3600, fiatcurrency: str='usd', fromdate: str='1578391200', todate: str='1599053802', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a balance history for the specified XPUB or address
		
		The value of sentToSelf is the amount sent from the same address to the same address or within addresses of xpub."
    addressorxpub: Address or XPUB
        blockchain: Blockchain name
        groupby: an interval in seconds, to group results by. Default is 3600 seconds
        fiatcurrency: if specified, the response will contain secondary (fiat) rate at the time of transaction. If not, all available currencies will be returned
        fromdate: specifies a start date as a Unix timestamp
        todate: specifies an end date as a Unix timestamp
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/balancehistory/{addressorxpub}"
    querystring = {}
    if groupby:
        querystring['groupBy'] = groupby
    if fiatcurrency:
        querystring['fiatcurrency'] = fiatcurrency
    if fromdate:
        querystring['fromDate'] = fromdate
    if todate:
        querystring['toDate'] = todate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tickers_v2(blockchain: str, timestamp: str='1519053802', currency: str='usd', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns currency rate for the specified currency and date. If the currency is not available for that specific timestamp, the next closest rate will be returned. All responses contain an actual rate timestamp."
    blockchain: Blockchain name
        timestamp: specifies a Unix timestamp to (/tickers-list) return available tickers for or (/tickers) that specifies a date to return currency rates for. If not specified, the last available rate will be returned.
        currency: specifies a currency of returned rate ("usd", "eur", "eth"...). If not specified, all available currencies will be returned
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/tickers"
    querystring = {}
    if timestamp:
        querystring['timestamp'] = timestamp
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_hash_v2(blockheight: int, blockchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get block hash by its height
		
		Note: Blockbook always follows the main chain of the backend it is attached to."
    blockheight: Block height/index
        blockchain: Blockchain name
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/block-index/{blockheight}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blockchain_info_summary(blockchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get basic summary of info relating to the currently selected blockchain"
    blockchain: Blockchain name
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_available_blockchains(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an array of active blockchains"
    
    """
    url = f"https://chain49.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_v2(blockchain: str, blockhashorheight: str, page: int=1, pagesize: int=1000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about block with transactions, subject to paging.
		
		Note: Blockbook always follows the main chain of the backend it is attached to. If there is a rollback-reorg in the backend, Blockbook will also do rollback. When you ask for block by height, you will always get the main chain block. If you ask for block by hash, you may get the block from another fork but it is not guaranteed (backend may not keep it)"
    blockchain: Blockchain name
        blockhashorheight: Block hash or height
        page: specifies page of returned transactions, starting from 1. If out of range, Blockbook returns the closest possible page.
        pagesize: number of transactions returned by call (default and maximum 1000)
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/block/{blockhashorheight}"
    querystring = {}
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mempool_v2(blockchain: str, pagesize: int=1000, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of transaction IDs currently in the mempool of the node (meaning unconfirmed transactions not included in any block yet)
		
		Note: this route was implemented by us and is therefore not yet supported by existing blockbook clients."
    blockchain: Blockchain name
        pagesize: number of transactions returned by call (default and maximum 1000)
        page: specifies page of returned transactions, starting from 1. If out of range, Blockbook returns the closest possible page.
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/mempool"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transaction_as_is_from_backend_v2(txid: str, blockchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns transaction data in the exact format as returned by backend, including all coin specific fields"
    txid: Transaction ID
        blockchain: Blockchain name
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/tx-specific/{txid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transaction_v2(blockchain: str, txid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transaction returns "normalized" data about transaction, which has the same general structure for all supported coins. It does not return coin specific fields (for example information about Zcash shielded addresses).
		
		A note about the blockTime field:
		for already mined transaction (confirmations > 0), the field blockTime contains time of the block
		for transactions in mempool (confirmations == 0), the field contains time when the running instance of Blockbook was first time notified about the transaction. This time may be different in different instances of Blockbook."
    blockchain: Blockchain name
        txid: Transaction ID
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/tx/{txid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_transaction_in_url_v2(hex: str, blockchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sends new transaction to backend
		
		It is recommended to use POST for sending transactions as there is a limit on how much data can be sent in the URL itself."
    hex: Transaction hex data
        blockchain: Blockchain name
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/sendtx/{hex}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tickers_list_v2(blockchain: str, timestamp: str='1519053802', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of available currency rate tickers (secondary currencies) for the specified date, along with an actual data timestamp."
    blockchain: Blockchain name
        timestamp: specifies a Unix timestamp to (/tickers-list) return available tickers for or (/tickers) that specifies a date to return currency rates for. If not specified, the last available rate will be returned.
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/v2/tickers-list"
    querystring = {}
    if timestamp:
        querystring['timestamp'] = timestamp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def json_rpc_over_http(blockchain: str, rpcparams: str, rpcmethod: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All JSON-RPC methods are also available as normal HTTP GET routes if you specify a method (and optional parameters) in the URL."
    blockchain: Blockchain name
        rpcparams: Optional: Parameters delimited by "/"
        rpcmethod: Method to execute on node
        
    """
    url = f"https://chain49.p.rapidapi.com/{blockchain}/rpc/{rpcmethod}/{rpcparams}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chain49.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

