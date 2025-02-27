import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def btcgetblockchaininfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Bitcoin Blockchain Information.</p>"
    
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xrpgetfee(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get XRP Blockchain fee.</p>"
    
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/xrp/v2/fee"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xrpgettransaction(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get XRP Transaction by transaction hash.</p>"
    hash: Transaction hash
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/xrp/v2/transaction/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xrpgetaccountinfo(account: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get XRP Account info.</p>"
    account: Account address
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/xrp/v2/account/{account}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btcgeneratewallet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Tatum supports BIP44 HD wallets. It is very convenient and secure, since it can generate 2^32-1 addresses from 1 mnemonic phrase. Mnemonic phrase consists of 12 special words in defined order and can restore access to all generated addresses and private keys.<br/>Each address is identified by 3 main values:<ul><li>Private Key - your secret value, which should never be revealed</li><li>Public Key - public address to be published</li><li>Derivation index - index of generated address</li></ul></p><p>Tatum follows BIP44 specification and generates for Bitcoin wallet with derivation path m'/44'/0'/0'/0. More about BIP44 HD wallets can be found here - <a target="_blank" href="https://github.com/bitcoin/bips/blob/master/bip-0044.mediawiki">https://github.com/bitcoin/bips/blob/master/bip-0044.mediawiki</a>.
		Generate BIP44 compatible Bitcoin wallet. For security and privacy reasons, this method should be called from local instance of Tatum Middleware, see <a href="https://github.com/tatumio/tatum-middleware">https://github.com/tatumio/tatum-middleware</a>.</p>
		"
    
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/wallet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethgetblock(hash: str, tx: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Ethereum Block by block hash or block number.</p>"
    hash: Block hash or block number
        tx: Enable transaction details:<ul><li> true - Transaction details included</li><li>false - Only transaction hashes present</li></ul>
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/block/{hash}/{tx}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xrpgetlastclosedledger(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get XRP Blockchain last closed ledger.</p>"
    
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/xrp/v2/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xrpwallet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Generate XRP account. For security and privacy reasons, this method should be called from local instance of Tatum Middleware, see <a href="https://github.com/tatumio/tatum-middleware">https://github.com/tatumio/tatum-middleware</a>.</p>"
    
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/xrp/v2/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btcgetblockhash(i: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Bitcoin Block hash. Returns hash of the block to get the block detail.</p>"
    i: The number of blocks preceding a particular block on a block chain.
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/block/hash/{i}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btcgetutxo(hash: str, index: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get UTXO Of given Transaction and output index. If index is not spent, data are returned, otherwise 404.</p>"
    hash: TX Hash
        index: Index of tx output to check if spent or not
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/utxo/{hash}/{index}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlibraaccount(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Libra Account state"
    address: Account Address
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/libra/v2/account/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlibraaccounttransaction(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Libra Account Transactions"
    address: Account Address
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/libra/v2/account/transaction/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethgenerateaddress(xpub: str, index: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Generate Ethereum account deposit address from Extended public key.</p>"
    xpub: Extended public key of wallet.
        index: Derivation index of desired address to be generated.
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/address/{xpub}/{index}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btcgetblock(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Bitcoin Block detail by block hash.</p>"
    hash: Block hash
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/block/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xrpgetaccounttx(account: str, min: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>List all Account transactions.</p>"
    account: Address of XRP account.
        min: Ledger version to start scanning for transactions from.
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/xrp/v2/account/tx/{account}"
    querystring = {}
    if min:
        querystring['min'] = min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethgettransactionreceipt(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Ethereum Transaction Receipt by transaction hash.</p>"
    hash: Transaction hash
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/transaction/{hash}/receipt"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethgetcurrentblock(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Ethereum Current Block number.</p>"
    
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/block/current"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethgettransaction(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Ethereum Transaction by transaction hash.</p>"
    hash: Transaction hash
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/transaction/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btcgenerateaddress(xpub: str, index: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Generate Bitcoin deposit address from Extended public key.</p>"
    xpub: Extended public key of wallet.
        index: Derivation index of desired address to be generated.
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/address/{xpub}/{index}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btcgetrawtransaction(hash: str, encode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Bitcoin Transaction by transaction hash. Encode Raw TX to JSON if desired.</p>"
    hash: Transaction hash
        encode: Encode Transaction to JSON<br/><ul><li> true - JSON output</li><li>false - RAW HEX output</li></ul>
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/transaction/{hash}/{encode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlibratransaction(start: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Libra Transactions"
    start: Start index
        limit: Limit of transactions
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/libra/v2/transaction/{start}/{limit}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btcgettxbyaddress(reverse: str, address: str, after: str='4c7846a8ff8415945e96937dea27bdb3144c15d793648d725602784826052586', limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Bitcoin Transaction by address.</p>"
    reverse: Reverse the order of transactions, false - from oldest to latest, true - from latest to oldest.
        address: Address
        after: A txid to include transactions after, this is often the last txid of a previous query.
        limit: The maximum number of results to return.
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/bitcoin/v2/transaction/address/{address}/{reverse}"
    querystring = {}
    if after:
        querystring['after'] = after
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethgeneratewallet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Tatum supports BIP44 HD wallets. It is very convenient and secure, since it can generate 2^32-1 addresses from 1 mnemonic phrase. Mnemonic phrase consists of 12 special words in defined order and can restore access to all generated addresses and private keys.<br/>Each address is identified by 3 main values:<ul><li>Private Key - your secret value, which should never be revealed</li><li>Public Key - public address to be published</li><li>Derivation index - index of generated address</li></ul></p><p>Tatum follows BIP44 specification and generates for Ethereum wallet with derivation path m'/44'/60'/0'/0. More about BIP44 HD wallets can be found here - <a target="_blank" href="https://github.com/bitcoin/bips/blob/master/bip-0044.mediawiki">https://github.com/bitcoin/bips/blob/master/bip-0044.mediawiki</a>.
		Generate BIP44 compatible Ethereum wallet. For security and privacy reasons, this method should be called from local instance of Tatum Middleware, see <a href="https://github.com/tatumio/tatum-middleware">https://github.com/tatumio/tatum-middleware</a>.</p>
		"
    
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/wallet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ethgetbalance(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Ethereum Account balance in ETH.</p>"
    address: Account address
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/account/balance/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etherc20getbalance(address: str, currency: str='USDT', contractaddress: str='0xdac17f958d2ee523a2206206994597c13d831ec7', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Get Ethereum ERC20 Account balance in smallest ERC20 unit.</p>"
    address: Account address
        currency: Tatum supported ERC20 currency. Either currency, or contractAddress must be present.
        contractaddress: ERC20 contract address to get balance of. Either contractAddress, or currency must be present.
        
    """
    url = f"https://tatum-blockchain-api-testnet1.p.rapidapi.com/ethereum/v2/account/balance/erc20/{address}"
    querystring = {}
    if currency:
        querystring['currency'] = currency
    if contractaddress:
        querystring['contractAddress'] = contractaddress
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tatum-blockchain-api-testnet1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

