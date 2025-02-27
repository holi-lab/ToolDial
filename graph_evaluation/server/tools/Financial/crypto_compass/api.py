import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news(exchange: str='Binance', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns latest news from all supported exchanges
		
		</div><h3><span>Response</span></h3>
		<table> 
		<thead>
		<tr>
		<th>Key</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		<td>date</td>
		<td>date of publication</td>
		</tr>
		<tr>
		<td>title</td>
		<td>title of announsment</td>
		</tr>
		<tr>
		<td>text</td>
		<td>text of announsment</td>
		</tr>
		
		
		<tr>
		<td>news_url</td>
		<td>url of original announsment</td>
		</tr>
		<tr>
		<tr>
		<td>source_name</td>
		<td>name of exchange that released announsment</td>
		</tr>
		</tbody>
		 </table>"
    
    """
    url = f"https://crypto-compass.p.rapidapi.com/news"
    querystring = {}
    if exchange:
        querystring['exchange'] = exchange
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_listings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns upcoming cryptocurrencies listings from the biggest exchanges
		
		</div><h3><span>Response</span></h3><span>
		</span><table> 
		<thead>
		<tr>
		<th>Key</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		<td>date</td>
		<td>date of publication</td>
		</tr>
		<tr>
		<td>title</td>
		<td>title of announsment</td>
		</tr>
		<tr>
		<td>text</td>
		<td>text of announsment</td>
		</tr>
		<tr>
		<td>topic</td>
		<td>specific topic(if set)</td>
		</tr>
		<tr>
		<td>ticker</td>
		<td>tiker of coin that's gonna be listed</td>
		</tr>
		<tr>
		<td>news_url</td>
		<td>url of original announsment</td>
		</tr>
		<tr>
		<td>news_type</td>
		<td>type of announsment(default "")</td>
		</tr>
		<tr>
		<td>sentiment</td>
		<td>default ""</td>
		</tr>
		<tr>
		<td>source_name</td>
		<td>name of exchange that released announsment</td>
		</tr>
		</tbody>
		 </table>"
    
    """
    url = f"https://crypto-compass.p.rapidapi.com/new_listings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_info(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts an exchange name as an input, returns a list of exchanges where the coin is traded, and prices for this coin
		<h3><span>Request</span></h3>
		<table> 
		<thead>
		<tr>
		<th>Key</th>
		<th>Required</th>
		<th>Value</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		<td>exchange</td>
		<td>required</td>
		<td>Binance</td>
		<td>name of exchange that you want to check</td>
		</tr>
		</tbody>
		 </table><span>
		</span></div><h3><span>Response</span></h3><span>
		</span><table> 
		<thead>
		<tr>
		<th>Key</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		<td>exchangeId</td>
		<td>unique identifier for exchange</td>
		</tr>
		<tr>
		<td>name</td>
		<td>proper name of exchange</td>
		</tr>
		<tr>
		<td>rank</td>
		<td>rank is in ascending order - this number is directly associated with the total exchange volume whereas the highest volume exchange receives rank 1</td>
		</tr>
		<tr>
		<td>percentTotalVolume</td>
		<td>the amount of daily volume a single exchange transacts in relation to total daily volume of all exchanges</td>
		</tr>
		<tr>
		<td>volumeUsd</td>
		<td>daily volume represented in USD</td>
		</tr>
		<tr>
		<td>tradingPairs</td>
		<td>number of trading pairs (or markets) offered by exchange</td>
		</tr>
		<tr>
		<td>socket</td>
		<td>true/false, true = trade socket available, false = trade socket unavailable</td>
		</tr>
		<tr>
		<td>exchangeUrl</td>
		<td>website to exchange</td>
		</tr>
		<tr>
		<td>updated</td>
		<td>UNIX timestamp (milliseconds) since information was received from this exchange</td>
		</tr>
		</tbody>
		 </table>"
    exchange: name of exchange that you want to check
        
    """
    url = f"https://crypto-compass.p.rapidapi.com/exchange_info"
    querystring = {'exchange': exchange, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_email(email: str, exchange: str, coin: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can send an email as a notification about a new listing using our pattern to be able to buy it before the price rase
		#### Response
		
		- status ok if letter sent else - error"
    
    """
    url = f"https://crypto-compass.p.rapidapi.com/send_email"
    querystring = {'email': email, 'exchange': exchange, 'coin': coin, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of exchanges from which we receive data"
    
    """
    url = f"https://crypto-compass.p.rapidapi.com/supported_exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscribe_for_email_notifications(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can send your email address and for new listings, it will send notifications, if there are no errors it will return the status: ok and send a test letter to the email address. Later you will receive an email if new coins start listing at supported exchanges
		
		#### Response
		
		- status ok if alert set else - error"
    
    """
    url = f"https://crypto-compass.p.rapidapi.com/set_alert"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asset_info(tiker: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts a coin ticker as an input, returns a list of exchanges where the coin is traded, and prices for this coin
		
		<h3><span>Request</span></h3>
		<table> 
		<thead>
		<tr>
		<th>Key</th>
		<th>Required</th>
		<th>Value</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		<td>asset</td>
		<td>required</td>
		<td>BTC</td>
		<td>tiker of coin that you want to check</td>
		</tr>
		</tbody>
		 </table><span>
		</span></div><h3><span>Response</span></h3><span>
		</span><table> 
		<thead>
		<tr>
		<th>Key</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		<td>priceUSD</td>
		<td>price for specific asset at specific exchange</td>
		</tr>
		<tr>
		<td>volumePercent</td>
		<td>percent of trading volume of specific asset at specific exchange</td>
		</tr>
		</tbody>"
    
    """
    url = f"https://crypto-compass.p.rapidapi.com/asset_info"
    querystring = {'tiker': tiker, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_pairs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns upcoming currency pair exits on all supported exchanges
		
		</div><h3><span>Response</span></h3><span>
		</span><table> 
		<thead>
		<tr>
		<th>Key</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		<td>date</td>
		<td>date of publication</td>
		</tr>
		<tr>
		<td>title</td>
		<td>title of announsment</td>
		</tr>
		<tr>
		<td>text</td>
		<td>text of announsment</td>
		</tr>
		<tr>
		<td>topic</td>
		<td>specific topic(if set)</td>
		</tr>
		<tr>
		<td>ticker</td>
		<td>tiker of trading pair that's gonna be added</td>
		</tr>
		<tr>
		<td>news_url</td>
		<td>url of original announsment</td>
		</tr>
		<tr>
		<td>news_type</td>
		<td>type of announsment(default "")</td>
		</tr>
		<tr>
		<td>sentiment</td>
		<td>extra parameter(default "")</td>
		</tr>
		<tr>
		<td>source_name</td>
		<td>name of exchange that released announsment</td>
		</tr>
		</tbody>
		 </table>"
    
    """
    url = f"https://crypto-compass.p.rapidapi.com/new_pairs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-compass.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

