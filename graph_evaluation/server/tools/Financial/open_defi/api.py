import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def web_pair_details(chain_slug: str, pair_slug: str, exchange_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    chain_slug: Blockchain slug, e.g, "ethereum"
        pair_slug: 
Trading pair friendly presentation (token0 symbol - token1 symbol) or pair pool contract address.

Symbols can be in any order: base token - quote token or quote token - base token.

If multiple trading pairs match the same symbol the one with the highest volume is returned
(assuming other trading pairs are ones with a fake token).

Examples of accepted `pair_slugs`:
- `ETH-USDC`
- `eth-usdc`
- `WETH-USDC`
- `USDC-WETH`
- `0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc`

The response is a composite of pair summary information, additional trading volume numbers and related links.

        exchange_slug: Exchange path slug e.g, `sushiswap`

        
    """
    url = f"https://open-defi.p.rapidapi.com/pair-details"
    querystring = {'chain_slug': chain_slug, 'pair_slug': pair_slug, 'exchange_slug': exchange_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_candles(time_bucket: str, pair_id: int, start: str='2020-05-18 00:00', end: str='2020-05-19 00:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "
		Fetch candle data for a single trading pair.
		
		You need to resolve the symbolic trading pair name to `pair_id` primary key with `/pairs` or `/pair-details`
		to call this API endpoint.
		
		Candle response size is limited to 10,000 candles. Only the first 10,000 candles
		since the start timestamp is returned.
		
		[Inspired by Bitfinex API](https://docs.bitfinex.com/reference#rest-public-candles).
		"
    time_bucket: 
What time bucket to use for the candle size.

        pair_id: 
Primary key for the trading pair

For tests and demos use id `1` which is ETH/USDC pair on Uniswap v2,
as it is the first deployed Uniswap v2 compatible pool.

        start: 
When the candle fetch period starts.

Use UNIX UTC timestamp, as ISO 8601 formatted string.
[Read more information about the time format](https://docs.python.org/3/library/datetime.html#datetime.datetime.fromisoformat).

If not specific this will be resolved to now() - time_bucket * 100, or getting the last 100 candles.

This range parameter is inclusive.

        end: 
When the candle fetch period ends.

Use UNIX UTC timestamp, as ISO 8601 formatted string.

If not specific this will be resolved to now().

This range parameter is inclusive.

        
    """
    url = f"https://open-defi.p.rapidapi.com/candles"
    querystring = {'time_bucket': time_bucket, 'pair_id': pair_id, }
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_pairs(direction: str='asc', page: int=0, filter_junk: bool=None, token0: str=None, page_size: int=20, sort: str='pair_id', token1: str=None, exchange_slugs: str="['uniswap-v2']", chain_slugs: str="['ethereum']", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "
		Query different combinations of the trading pairs.
		
		The most common query is all trading pairs of a decentralised exchange.
		
		`/pairs?chain_slugs=ethereum&exchange-slugs=uniswap-v2`
		
		The results are always paginated.
		
		## Parameter seralisation
		
		Note that lists are not JSON serialised, but use [OpenAPI query parameter serialization](https://swagger.io/docs/specification/serialization/). I.e. lists are comma separated, no spaces between items.
		"
    direction: Sort order:
 * `asc` - Ascending, from A to Z
 * `desc` - Descending, from Z to A

        page: Page number, zero-indexed
        filter_junk: If there are multiple token pairs with matching symbols in the result,
choose one with the highest volume and hide the others.

        token0: Symbol of one of the token pairs e.g. WETH
        page_size: Limit number of pairs returned (default is 20)
        sort: 
What kind of sort options is available for the list

The default `pair_id` sorting makes only sense for machine-to-machine actions.
The recommended sort method is by descending volume (most interesting trading pairs first).

        token1: Symbol of one of the token pairs e.g. USDC
        exchange_slugs: Slugs for DEXes to look up
        chain_slugs: Slugs for blockchain to look up
        
    """
    url = f"https://open-defi.p.rapidapi.com/pairs"
    querystring = {}
    if direction:
        querystring['direction'] = direction
    if page:
        querystring['page'] = page
    if filter_junk:
        querystring['filter_junk'] = filter_junk
    if token0:
        querystring['token0'] = token0
    if page_size:
        querystring['page_size'] = page_size
    if sort:
        querystring['sort'] = sort
    if token1:
        querystring['token1'] = token1
    if exchange_slugs:
        querystring['exchange_slugs'] = exchange_slugs
    if chain_slugs:
        querystring['chain_slugs'] = chain_slugs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_pair_trade_data(pair_id: int, period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pair_id: 
Pair id as received in `/pairs` or `/pair-details`

As the endpoint /pair-details is preferred, this endpoint only gives direct pair_id access for now.

        period: 
The time perid for the stats.

Currently only latest stats are supported.

        
    """
    url = f"https://open-defi.p.rapidapi.com/pair-trade-data"
    querystring = {'pair_id': pair_id, 'period': period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_exchange_details(exchange_slug: str, chain_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    exchange_slug: Exchange path slug e.g, `sushiswap`

        chain_slug: Blockchain slug, e.g, "ethereum"
        
    """
    url = f"https://open-defi.p.rapidapi.com/exchange-details"
    querystring = {'exchange_slug': exchange_slug, 'chain_slug': chain_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_exchanges(sort: str='usd_volume_30d', direction: str='desc', filter_zero_volume: bool=None, chain_slug: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    sort: 
What kind of sort options is available for the output.

The default `usd_volume_30d` sorts the exchanges based on their monthly volume.

        direction: Sort order:
 * `asc` - Ascending, from A to Z
 * `desc` - Descending, from Z to A

        filter_zero_volume: When set, do not return exchanegs which have had no volume for last 30 days.

This will speed up the response a bit, as most exchanges are ghost exchanges.

        chain_slug: Blockchain slug, e.g, `ethereum` for Ethereum mainnet.

If present, list exchanges only for this chain.

        
    """
    url = f"https://open-defi.p.rapidapi.com/exchanges"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if direction:
        querystring['direction'] = direction
    if filter_zero_volume:
        querystring['filter_zero_volume'] = filter_zero_volume
    if chain_slug:
        querystring['chain_slug'] = chain_slug
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_impressive_numbers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://open-defi.p.rapidapi.com/impressive-numbers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_chains(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://open-defi.p.rapidapi.com/chains"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_chain_details(chain_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    chain_slug: Blockchain slug, e.g, `ethereum` for Ethereum mainnet

        
    """
    url = f"https://open-defi.p.rapidapi.com/chain-details"
    querystring = {'chain_slug': chain_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_top_momentum(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "
		Return top up and down 100 price changes for 24h.
		
		The results are split for all trading pairs and trading pairs with $1M min liquidity.
		"
    
    """
    url = f"https://open-defi.p.rapidapi.com/top-momentum"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-defi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

