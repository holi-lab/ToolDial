import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def chains(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all chains supported by us with the projects."
    
    """
    url = f"https://defimercury.p.rapidapi.com/chains"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defimercury.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def projects(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all projects currently supported by us with their chains"
    
    """
    url = f"https://defimercury.p.rapidapi.com/projects"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defimercury.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pool(pooladdress: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data on a pool.
		
		**underlyingTokens**: An array of underlying token strings.
		
		**rewardTokens**: An array of reward token strings.
		
		**updatedTimestamp**: The Unix timestamp (in seconds) when the information about the liquidity pool was last updated.
		
		**pool**: The address of the liquidity pool on the Ethereum blockchain, in hexadecimal format (with a "0x" prefix) with chain.
		
		**chain**: The blockchain network where the liquidity pool is deployed (e.g., "ethereum", "polygon", "fantom").
		
		**symbol**: The symbol of the liquidity pool token.
		
		**tvlUsd**: The total value locked (TVL) in the liquidity pool, in USD.
		
		**apyBase**: The annual percentage yield (APY) for the underlying token(s) in the liquidity pool, expressed as a decimal.
		
		**apyReward**: The APY for the reward token(s) in the liquidity pool, expressed as a decimal.
		
		**apyTotal**: The total APY for the liquidity pool, including both underlying and reward tokens, expressed as a decimal.
		
		**url**: A link to the interface for the liquidity pool.
		
		**apyBaseBorrow**: The APY for borrowing the underlying token(s) in the liquidity pool, expressed as a decimal.
		
		**totalSupplyUsd**: The total value (in USD) of the liquidity pool token(s) in circulation.
		
		**totalBorrowUsd**: The total value (in USD) of the underlying token(s) that have been borrowed from the liquidity pool.
		
		**dailyAvgTvl**: The daily average TVL in the liquidity pool in USD.
		
		**dailyAvgApyBase**: The daily average APY for the underlying token(s) in the liquidity pool, expressed as a decimal.
		
		**dailyAvgApyReward**: The daily average APY for the reward token(s) in the liquidity pool, expressed as a decimal.
		
		**dailyAvgApyTotal**: The daily average total APY for the liquidity pool, including both underlying and reward tokens, expressed as a decimal."
    pooladdress: The address of the liquidity pool, in hexadecimal format (with a 0x prefix) with chain separated by -
        
    """
    url = f"https://defimercury.p.rapidapi.com/pool"
    querystring = {'poolAddress': pooladdress, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defimercury.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pools(showzerotvl: bool=None, project: str='aave-v3', page: int=1, minapy: int=0, limit: int=20, mintvl: int=1, chain: str='ethereum', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch collection of pools."
    showzerotvl: This parameter is a boolean value that determines whether or not to show liquidity pools with zero total value locked (TVL). If set to true, then all liquidity pools will be displayed, regardless of whether they have TVL or not. If set to false, then only liquidity pools with non-zero TVL will be displayed.
        project: This parameter specifies the name of the project. It is used to filter the results and only show liquidity pools that belong to this particular project.
        page: This parameter specifies the page number of the results. The API returns results in pages, with each page containing a specified number of liquidity pools. This parameter is used to navigate between pages and retrieve more results.
        minapy: This parameter is used to filter the results by the minimum annual percentage yield (APY) for the liquidity pools. Only liquidity pools with APY greater than or equal to the specified value will be returned.
        limit: This parameter specifies the number of liquidity pools to return per page. The API will return up to this number of liquidity pools per page. 
20 by default
50 is the max limit.
        mintvl: This parameter is used to filter the results by the minimum total value locked. Only liquidity pools with TVL greater than or equal to the specified value will be returned.
        chain: This parameter is used to filter the results by the blockchain network on which the liquidity pools are deployed. 
        
    """
    url = f"https://defimercury.p.rapidapi.com/pools"
    querystring = {}
    if showzerotvl:
        querystring['showZeroTVL'] = showzerotvl
    if project:
        querystring['project'] = project
    if page:
        querystring['page'] = page
    if minapy:
        querystring['minAPY'] = minapy
    if limit:
        querystring['limit'] = limit
    if mintvl:
        querystring['minTVL'] = mintvl
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defimercury.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

