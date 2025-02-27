import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettime(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the server time. This endpoint is also useful as a method for a client to 'ping' the Stronghold Platform and measure latency."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/utilities/time"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listmarkets(venueid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of the well known markets on the venue."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/markets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassets(venueid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of known assets for the current venue.
		
		## Response
		
		Field | Type | Description
		--- | --- | ---
		Result | Asset[] | List of known assets for the current venue."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/assets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listaccounttrades(venueid: str, accountid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of the account's most recently executed trades."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/accounts/{accountid}/trades"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarketorderbook(venueid: str, marketid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the orderbook for a given market.
		
		## Response
		
		Field | Type | Description
		--- | --- | ---
		result | Orderbook | The market orderbook.
		result.bids | string[][] | The bid orders.
		result.asks | string[][] | The ask orders.
		
		Each item in the bids or asks arrays is an array describing a particular price with a size: ["(price)", "(size)"]."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/markets/{marketid}/orderbook"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generateauuid(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate an RFC 4122 v4 UUID."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/utilities/uuid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listmarkettrades(venueid: str, marketid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of historic trades for a given market. The last 200 trades are provided by default. If your use case requires more history, please contact developers@stronghold.co
		
		## Response
		
		Field | Type | Description
		--- | --- | ---
		result | Object | The latest historic trades.
		result.marketId | string | The market's identifier.
		result.trades | string[][] | The trades.
		
		Each item in the trades array is an array describing a particular trade: ["(price)", "(size)", "(time)"]."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/markets/{marketid}/trades"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccount(accountid: str, venueid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve account information, including balances.
		
		The result will contain the following information:
		- **ID**: the account's identifier. Stronghold associates a protocol-agnostic identifier that is used in preference to the 'native' ID, such as a 'public address'.
		- **Balances**: the account's balances. Asset information will be provided where possible."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/accounts/{accountid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccountorders(accountid: str, venueid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the open orders for an account."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/accounts/{accountid}/orders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getavailablevenues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the venues the credential has access to.
		
		Each venue describes its:
		- **capabilities**: as each venue differs in protocol and context, not all Platform APIs are supported on all venues. The capabilities object describes which operations can be performed.
		- **attributes**: protocol/context-specific key-value pairs."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccounttransactions(accountid: str, venueid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Response
		
		Field | Type | Description
		--- | --- | ---
		result | AccountTransaction[] | The account's transactions."
    
    """
    url = f"https://testingapiwithinorg.p.rapidapi.com/venues/{venueid}/accounts/{accountid}/transactions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testingapiwithinorg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

