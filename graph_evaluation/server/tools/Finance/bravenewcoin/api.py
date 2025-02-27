import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dailyohlcv(timestamp: str=None, size: int=10, startafter: str=None, indextype: str=None, indexid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns end-of-day open, high, low, close prices and volumes (OHLCV) for market weighted averages (MWA) or global weighted averages (GWA). The 24hr period for end of day is based on an open/close at 00:00:00 UTC. Parameters may be used to select the data required. 
		
		BNC uses universal identifiers (UUIDs) for assets and markets. Each index type will have a UUID which identifies the constituent type used to build the index:
		
		| Index Type | Index Id  | Output |
		| -----------|:---------:|:---------:|
		| MWA        | Market Id | OHLCV for that market |
		| GWA        | Asset Id  | OHLCV for that coin / asset |
		
		The UUIDs for assets or markets may be found from the Lookups with those names.
		
		If no timestamp is provided, then only the latest values are returned. With a timestamp, records are returned in reverse date order, the number determining how far back  controlled by the size parameter. 
		
		An authentication token is required. Obtain a token valid for 24 hours from the GetToken endpoint."
    timestamp: Retrieve all daily OHLCV records up to the timestamp provided. All dates are stored in UTC. Timestamp strings should be in the form  YYYY-MM-DDThh:mm:ssZ
        size: Integer 1 to 2000. Default 10. The maximum size to return in the result set up to the overall limit of 2000. This parameter is ignored if no timestamp is provided. Otherwise, since records are in reverse data order, use in conjunction with timestamp to make selections back in time.
        startafter: UUID. Retrieve OHLCV for the indexId or indexType required, starting after this particular record id. Every record in the dataset has a universal identifier (UUID). This parameter provides for pagination through large selections since every response includes a nextId that can be used here.
        indextype: Restrict the OHLCV results to the index type. Either MWA or GWA.
        indexid: Retrieve all the OHLCV values for a particular asset or market by its id. The index id is a universal identifier (UUID) that will differ based on the index type.

| Index Type | Index Id  |
| -----------|:---------:|
| MWA        |Market Id  |
| GWA        |Asset Id   |

UUIDs for assets or markets can be identified from their endpoints in Lookups.
        
    """
    url = f"https://bravenewcoin.p.rapidapi.com/ohlcv"
    querystring = {}
    if timestamp:
        querystring['timestamp'] = timestamp
    if size:
        querystring['size'] = size
    if startafter:
        querystring['startAfter'] = startafter
    if indextype:
        querystring['indexType'] = indextype
    if indexid:
        querystring['indexId'] = indexid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bravenewcoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def assetbyid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the details of an individual asset. Supply the unique identifier (UUID) of the asset.
		
		The full list of all asset details can be obtained from the Asset endpoint."
    id: The unique resource identifier (UUID) of an asset e.g.  BTC = f1ff77b6-3ab4-4719-9ded-2fc7e71cff1f
        
    """
    url = f"https://bravenewcoin.p.rapidapi.com/asset/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bravenewcoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def marketbyid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the details of an individual market. Supply the unique identifier (UUID) of the market.
		
		The full list of all market UUIDs can be obtained from the Market endpoint."
    id: UUID. The unique resource identifier of a market e.g.  BTC/USD = 6ea0d2ef-6dd0-4adb-ad32-f7f3db58ccbe
        
    """
    url = f"https://bravenewcoin.p.rapidapi.com/market/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bravenewcoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asset(status: str='ACTIVE', type: str=None, symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the UUID and details for all assets or provide an optional query parameter to search.
		Assets may have a status . Active assets will be available at both AssetTicker and DailyOHLCV endpoints. Inactive assets are not available for AssetTickers but may provide some historical OHLCV data if available."
    status: ACTIVE (default) or INACTIVE.  Only return assets which have the particular status. 
        type: CRYPTO or FIAT. Only return assets of the particular type. 
        symbol: Only return assets which have a particular ticker symbol e.g. BTC.
        
    """
    url = f"https://bravenewcoin.p.rapidapi.com/asset"
    querystring = {}
    if status:
        querystring['status'] = status
    if type:
        querystring['type'] = type
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bravenewcoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market(quoteassetid: str=None, baseassetid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the UUIDs for all markets or provide a query parameter to search.
		
		The parameters are asset UUIDs. The full list of all asset details is available from the Asset endpoint."
    quoteassetid: UUID. Only return markets which contain the asset id on the quote side of the market. Supply the UUID of an asset e.g.  USD = e77b9824-126a-418e-a69c-a2e682578e94
        baseassetid: UUID. Only return markets which contain the asset id on the base side of the market. Supply the UUID of an asset e.g.  BTC = f1ff77b6-3ab4-4719-9ded-2fc7e71cff1f
        
    """
    url = f"https://bravenewcoin.p.rapidapi.com/market"
    querystring = {}
    if quoteassetid:
        querystring['quoteAssetId'] = quoteassetid
    if baseassetid:
        querystring['baseAssetId'] = baseassetid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bravenewcoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def assetticker(assetid: str, percentchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This free endpoint will return the latest details for an asset refreshed every 5 minutes with global USD spot price, 24 hour volume and supply.  The optional percentChange parameter provides the 1, 7 and 30 day percentage movements as additional payload.   
		
		Supply the unique identifier (UUID) for an asset.  The UUID for any asset may be found from the Assets endpoint in Lookups.
		
		A security token is required for the Authorization Header. Obtain a token valid for 24 hours from the GetToken endpoint."
    assetid: UUID. The unique resource identifier of an asset e.g.  BTC = f1ff77b6-3ab4-4719-9ded-2fc7e71cff1f
        percentchange: true or false (default).  When true the  percentage changes in the price and 24 hour volume across 1, 7 and 30 days will be included in the response payload. 
        
    """
    url = f"https://bravenewcoin.p.rapidapi.com/market-cap"
    querystring = {'assetId': assetid, }
    if percentchange:
        querystring['percentChange'] = percentchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bravenewcoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

