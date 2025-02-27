import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def swap(walletaddress: str, blockchainid: str, totokenaddress: str, amounttrade: str, fromtokenaddress: str, slippage: int, gasprice: str='12500000000', gaslimit: int=11500000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate data for calling the router for exchange. The data ready to send the transaction."
    walletaddress: The address of the wallet that calls the exchange router.
        blockchainid: The blockchain id you can choose from eth, bsc, polygon, avalanche, optimism, arbitrum, gnosis.
        totokenaddress: The address of the requested token.
        amounttrade: The amount of tokens to trade in wei.
        fromtokenaddress: The address of the trade in token.
        slippage: Limit of price slippage you are willing to accept in percentage, may be set with decimals. 0.5 means 0.5% slippage is acceptable. Low values increase chances that transaction will fail, high values increase chances of front running. Set values in the range from 0 to 50.
        gasprice: The router takes in account gas expenses to determine exchange route. It is important to use the same gas price on the quote and swap methods.Gas price set in wei 12.5 GWEI set as 12500000000.
        gaslimit: Maximum amount of gas for a swap. Should be the same for a quote and swap. Default value 11500000
        
    """
    url = f"https://de-fi-crypto-currency-exchange.p.rapidapi.com/swap/{blockchainid}"
    querystring = {'walletAddress': walletaddress, 'toTokenAddress': totokenaddress, 'amountTrade': amounttrade, 'fromTokenAddress': fromtokenaddress, 'slippage': slippage, }
    if gasprice:
        querystring['gasPrice'] = gasprice
    if gaslimit:
        querystring['gaslimit'] = gaslimit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "de-fi-crypto-currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rate(blockchainid: str, totokenaddress: str, amount: str, fromtokenaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the best quote to exchange."
    blockchainid: The blockchain id you can choose from eth, bsc, polygon, avalanche, optimism, arbitrum, gnosis.
        totokenaddress: The address of the requested token.
        amount: The amount of tokens to trade in wei.
        fromtokenaddress: The address of the trade in token.
        
    """
    url = f"https://de-fi-crypto-currency-exchange.p.rapidapi.com/rate/{blockchainid}"
    querystring = {'toTokenAddress': totokenaddress, 'amount': amount, 'fromTokenAddress': fromtokenaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "de-fi-crypto-currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def approve(blockchainid: str, amount: str, tokenaddress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate data for calling the contract in order to allow the router to spend funds. The data ready to send the approval transaction."
    blockchainid: The blockchain id you can choose from eth, bsc, polygon, avalanche, optimism, arbitrum, gnosis.
        amount: The number of tokens that the router is allowed to spend in wei. If not specified, it will be allowed to spend an infinite amount of tokens.
        tokenaddress: The address of the token you want to exchange.


        
    """
    url = f"https://de-fi-crypto-currency-exchange.p.rapidapi.com/approve/{blockchainid}"
    querystring = {'amount': amount, 'tokenAddress': tokenaddress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "de-fi-crypto-currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tokens(blockchainid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of tokens available for swap in the selected blockchain. All supported tokens (can also use your own)."
    blockchainid: The blockchain id you can choose from eth, bsc, polygon, avalanche, optimism, arbitrum, gnosis.
        
    """
    url = f"https://de-fi-crypto-currency-exchange.p.rapidapi.com/tokens/{blockchainid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "de-fi-crypto-currency-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

