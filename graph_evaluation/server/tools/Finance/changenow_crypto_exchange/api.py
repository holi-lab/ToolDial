import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchange_range_fixed_rate(from_to: str, api_key: str, userateid: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>New!</b>
		
		The API endpoint returns minimal payment amount and maximum payment amount required to make an exchange. If you try to exchange less than minimum or more than maximum, the transaction will most likely fail. Any pair of assets has minimum amount and some of pairs have maximum amount.
		
		<h3>Successful response:</h3>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>minAmount</i></b></td>
		        <td><i>Number</i></td>
		        <td>Minimal payment amount</td>
		    </tr>
		    <tr>
		        <td><b><i>maxAmount</i></b></td>
		        <td><i>Number|null</i></td>
		        <td>Maximum payment amount. Could be null.</td>
		    </tr>
		</table>
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    from_to: (Required) Underscore separated pair of tickers
        userateid: (Optional) Use rateId for fixed-rate flow. If this field is true, you could use returned field \\\\\\\"rateId\\\\\\\" in next method for creating transaction to freeze estimated amount that you got in this method. Current estimated amount would be valid until time in field \\\\\\\"validUntil\\\\\\\"
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v1/exchange-range/fixed-rate/{from_to}"
    querystring = {'api_key': api_key, }
    if userateid:
        querystring['useRateId'] = userateid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_exchange_range(fromcurrency: str, tocurrency: str, tonetwork: str='eth', flow: str='standard', fromnetwork: str='btc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API endpoint returns minimal payment amount and maximum payment amount required to make an exchange. If you try to exchange less than minimum or more than maximum, the transaction will most likely fail. Any pair of assets has minimum amount and some of pairs have maximum amount.
		
		<h3>Successful response:</h3>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>fromCurrency</i></b></td>
		        <td><i>String</i></td>
		        <td>Ticker of the currency you want to exchange</td>
		    </tr>
		    <tr>
		        <td><b><i>fromNetwork</i></b></td>
		        <td><i>String</i></td>
		        <td>Network of the currency you want to exchange</td>
		    </tr>
		    <tr>
		        <td><b><i>toCurrency</i></b></td>
		        <td><i>String</i></td>
		        <td>Ticker of the currency you want to receive</td>
		    </tr>
		    <tr>
		        <td><b><i>toNetwork</i></b></td>
		        <td><i>String</i></td>
		        <td>Network of the currency you want to receive</td>
		    </tr>
		    <tr>
		        <td><b><i>flow</i></b></td>
		        <td><i>String</i></td>
		        <td>Type of exchange flow. Enum: ["standard", "fixed-rate"]</td>
		    </tr>
		    <tr>
		        <td><b><i>minAmount</i></b></td>
		        <td><i>Number</i></td>
		        <td>Minimal payment amount</td>
		    </tr>
		    <tr>
		        <td><b><i>maxAmount</i></b></td>
		        <td><i>Number|null</i></td>
		        <td>Maximum payment amount. Could be null.</td>
		    </tr>
		</table>
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    fromcurrency: (Required) Ticker of the currency you want to exchange
        tocurrency: (Required) Ticker of the currency you want to receive
        tonetwork: (Optional) Network of the currency you want to receive
        flow: (Optional) Type of exchange flow. Enum: [\\\"standard\\\", \\\"fixed-rate\\\"]. Default value is standard
        fromnetwork: (Optional) Network of the currency you want to exchange
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/exchange/range"
    querystring = {'fromCurrency': fromcurrency, 'toCurrency': tocurrency, }
    if tonetwork:
        querystring['toNetwork'] = tonetwork
    if flow:
        querystring['flow'] = flow
    if fromnetwork:
        querystring['fromNetwork'] = fromnetwork
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_list_of_available_currencies(flow: str='standard', active: str=None, sell: str=None, buy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns the list of available currencies.
		
		<h3>Successful response:</h3>
			<p>The response contains an array of objects with currency information.</p>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>ticker</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency ticker</td>
		    </tr>
		    <tr>
		        <td><b><i>name</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency name</td>
		    </tr>
		    <tr>
		        <td><b><i>image</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency logo url</td>
		    </tr>
		    <tr>
		        <td><b><i>hasExternalId</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency has an Extra ID</td>
		    </tr>
		    <tr>
		        <td><b><i>isFiat</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is a fiat currency (EUR, USD)</td>
		    </tr>
		    <tr>
		        <td><b><i>featured</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is popular</td>
		    </tr>
		    <tr>
		        <td><b><i>isStable</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is stable</td>
		    </tr>
		    <tr>
		        <td><b><i>supportsFixedRate</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is available on a fixed-rate flow</td>
		    </tr>
		    <tr>
		        <td><b><i>network</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency network</td>
		    </tr>
		    <tr>
		        <td><b><i>tokenContract</i></b></td>
		        <td><i>String</i></td>
		        <td>Contract for token or null for non-token</td>
		    </tr>
		    <tr>
		        <td><b><i>buy</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is available to buy</td>
		    </tr>
		    <tr>
		        <td><b><i>sell</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is available to sell</td>
		    </tr>
		</table>
		
		<h3>Request Parameters:</h3>"
    flow: (Optional) Type of exchange flow. Enum: [\"standard\", \"fixed-rate\"]. Default value is standard
        active: (Optional) Set true to return only active currencies
        sell: (Optional) If this field is true, only currencies available for sell are returned.
        buy: (Optional) If this field is true, only currencies available for buy are returned.
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/exchange/currencies"
    querystring = {}
    if flow:
        querystring['flow'] = flow
    if active:
        querystring['active'] = active
    if sell:
        querystring['sell'] = sell
    if buy:
        querystring['buy'] = buy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_address_validation(currency: str, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint validates the address with a checksum depending on a transferred network.
		The ‘result’ field in the response shows if the address is valid: True if valid, and False if the address is invalid. The ‘message’ field describes why the address is invalid. In case the address is valid, this field is null.
		An error may occur in case the request is incorrect: a cryptocurrency or an address are not indicated, or we do not support this network yet.
		
		<h3>Successful response:</h3>
		The response contains the ‘result’ and ‘message’ fields.
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>result</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>The validity of an address</td>
		    </tr>
		    <tr>
		        <td><b><i>message</i></b></td>
		        <td><i>String|null</i></td>
		        <td>Explains why the address is invalid</td>
		    </tr>
		</table>
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    currency: (Required) The network of the address
        address: (Required) Address for validation
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/validate/address"
    querystring = {'currency': currency, 'address': address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_market_estimate_fiat_crypto_to_crypto(tocurrency: str, fromcurrency: str, type: str='direct', fromamount: int=1000, toamount: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint provides the direct and reverse market crypto-to-crypto or fiat-to-crypto estimated amounts.  
		<b>Attention! Do not use this endpoint for financial aims, only for informational! These rates don't include any fees.</b>  
		To work with this endpoint, provide your API key in the X-CHANGENOW-API-KEY title.  
		To calculate the direct estimated amount, set: fromCurrency, toCurrency, fromAmount, type: direct  
		To calculate the reverse estimated amount, set: fromCurrency, toCurrency, toAmount, type: reverse  
		
		<h3>Successful response:</h3>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>fromCurrency</i></b></td>
		        <td><i>String</i></td>
		        <td>“From” currency</td>
		    </tr>
		    <tr>
		        <td><b><i>toCurrency</i></b></td>
		        <td><i>String</i></td>
		        <td>“To” currency</td>
		    </tr>
		    <tr>
		        <td><b><i>fromAmount</i></b></td>
		        <td><i>Number</i></td>
		        <td>The amount of “from” currency</td>
		    </tr>
		    <tr>
		        <td><b><i>toAmount</i></b></td>
		        <td><i>Number</i></td>
		        <td>The amount of “to” currency</td>
		    </tr>
		    <tr>
		        <td><b><i>type</i></b></td>
		        <td><i>String</i></td>
		        <td>The type of the estimated amount — direct or reverse</td>
		    </tr>
		</table>
		
		<h3>Request Parameters:</h3>"
    tocurrency: (Required) \\\\\\\\\\\\\\\"To\\\\\\\\\\\\\\\" currency
        fromcurrency: (Required) \\\\\\\\\\\\\\\"From\\\\\\\\\\\\\\\" currency
        type: (Optional) Valid values: [direct, reverse] 
If the type is not set, ‘direct’ is used by default.
        fromamount: (Optional)  Set if this is a direct type of the estimated amount
        toamount: (Optional) Set if this is a reverse type of the estimated amount
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/markets/estimate"
    querystring = {'toCurrency': tocurrency, 'fromCurrency': fromcurrency, }
    if type:
        querystring['type'] = type
    if fromamount:
        querystring['fromAmount'] = fromamount
    if toamount:
        querystring['toAmount'] = toamount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_estimated_exchange_network_fee(fromnetwork: str, tocurrency: str, fromcurrency: str, convertedcurrency: str, convertednetwork: str, tonetwork: str, fromamount: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an estimated value that will be spent on paying network fees during an exchange.
		
		This number is ALREADY included in the estimate.
		
		## SUCCESSFUL RESPONSE:
		The response contains the ‘estimatedFee’ object and 'deposit', 'withdrawal', 'totals', and 'converted' fields inside it.
		
		
		
		### SUCCESSFUL RESPONSE FIELDS
		
		<table>
		    <tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>estimatedFee</i></b></td>
		        <td><i>Object</i></td>
		        <td>Object that contains detailed info on the network fee estimation.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>deposit</i></b></td>
		        <td><i>Object</i></td>
		        <td>Object that contains detailed info on the deposit network fees.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>currency</i></b></td>
		        <td><i>String</i></td>
		        <td>Deposit currency's ticker.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>network</i></b></td>
		        <td><i>String</i></td>
		        <td>Deposit currency's network.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>amount</i></b></td>
		        <td><i>Number</i></td>
		        <td>Network fee in the deposit currency.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>withdrawal</i></b></td>
		        <td><i>Object</i></td>
		        <td>Object that contains detailed info on the withdrawal network fees.
		        </td>
		    </tr>
		<tr>
		        <td><b><i>currency</i></b></td>
		        <td><i>String</i></td>
		        <td>Withdrawal currency's ticker.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>network</i></b></td>
		        <td><i>String</i></td>
		        <td>Withdrawal currency's network.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>amount</i></b></td>
		        <td><i>Number</i></td>
		        <td>Network fee in the withdrawal currency.
		        </td>
		    </tr>
		<tr>
		        <td><b><i>totals</i></b></td>
		        <td><i>Object</i></td>
		        <td>Object that contains combined network fee in deposit or withdeawal currency. 
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>from</i></b></td>
		        <td><i>Object</i></td>
		        <td>Object that contains combined network fee estimated to the deposit currency.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>to</i></b></td>
		        <td><i>Object</i></td>
		        <td>Object that contains combined network fee estimated to the withdrawal currency.        </td>
		    </tr>
		    <tr>
		        <td><b><i>converted</i></b></td>
		        <td><i>Object</i></td>
		        <td>Object that contains detailed info on the network fee estimation in select currency.        </td>
		    </tr>
		    <tr>
		        <td><b><i>currency</i></b></td>
		        <td><i>String</i></td>
		        <td>Network fee currency's ticker.        </td>
		    </tr>
		    <tr>
		        <td><b><i>network</i></b></td>
		        <td><i>String</i></td>
		        <td>Network of currency's ticker.        </td>
		    </tr>
		    <tr>
		        <td><b><i>deposit</i></b></td>
		        <td><i>Number</i></td>
		        <td>Deposit fee in the selected currency.
		        </td>
		    </tr>
		    <tr>
		        <td><b><i>withdrawal</i></b></td>
		        <td><i>Number</i></td>
		        <td>Withdrawal fee in the selected currency.
		        </td>
		    </tr>
		<tr>
		        <td><b><i>total</i></b></td>
		        <td><i>Number</i></td>
		        <td>Combined network fee in selected currency.
		        </td>
		    </tr>
		<tr>"
    fromnetwork: (Optional) Used to disambiguate multichain currencies.
        tocurrency: (Required) Ticker of the currency you want to receive
        fromcurrency: (Required) Ticker of the currency you want to exchange
        convertedcurrency: (Optional) Ticker of the currency you want to convert
        convertednetwork: (Optional) Used to disambiguate multichain currencies.
        tonetwork: (Optional) Used to disambiguate multichain currencies.
        fromamount: (Required if type is direct) Must be greater then 0.
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/exchange/network-fee"
    querystring = {'fromNetwork': fromnetwork, 'toCurrency': tocurrency, 'fromCurrency': fromcurrency, 'convertedCurrency': convertedcurrency, 'convertedNetwork': convertednetwork, 'toNetwork': tonetwork, 'fromAmount': fromamount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_user_addresses(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns a list of addresses bound to address name.
		 
		<h3>Successful response:</h3>
			<p>The response contains an array of addresses bound to address name.</p>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>success</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a request was processed successfully</td>
		    </tr>
		    <tr>
		        <td><b><i>addresses</i></b></td>
		        <td><i>Array</i></td>
		        <td>Array of addresses for requested fio-address or unstoppable-domain</td>
		    </tr>
		    <tr>
		        <td><b><i>currency</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency ticker in naming space of his protocol</td>
		    </tr>
		    <tr>
		        <td><b><i>chain</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency chain in naming space of his protocol</td>
		    </tr>
		    <tr>
		        <td><b><i>address</i></b></td>
		        <td><i>String</i></td>
		        <td>Real address for requested fio-address or unstoppable-domain</td>
		    </tr>
		    <tr>
		        <td><b><i>protocol</i></b></td>
		        <td><i>String</i></td>
		        <td>Protocol of current address</td>
		    </tr>
		</table>
		
		<h3>Request Parameters:</h3>"
    name: (Required) FIO address or Unstoppable domain as name.zil / name.crypto
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/addresses-by-name"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_minimal_exchange_amount(flow: str, tocurrency: str, fromcurrency: str, fromnetwork: str, tonetwork: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API endpoint returns minimal payment amount required to make an exchange. If you try to exchange less, the transaction will most likely fail.
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    flow: (Optional) Type of exchange flow. Enum: [\"standard\", \"fixed-rate\"]. Default value is standard
        tocurrency: (Required) Ticker of the currency you want to receive
        fromcurrency: (Required) Ticker of the currency you want to exchange
        fromnetwork: (Optional) Network of the currency you want to exchange
        tonetwork: (Optional) Network of the currency you want to receive
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/exchange/min-amount"
    querystring = {'flow': flow, 'toCurrency': tocurrency, 'fromCurrency': fromcurrency, 'fromNetwork': fromnetwork, 'toNetwork': tonetwork, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_estimated_exchange_amount(fromamount: int, flow: str='fixed-rate', tocurrency: str='usdt', userateid: str=None, type: str=None, fromnetwork: str='btc', tonetwork: str='eth', toamount: str=None, fromcurrency: str='btc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns estimated exchange amount for the exchange and some additional fields. Accepts to and from currencies, currencies' networks, exchange flow, and RateID.
		
		<h3>Successful response:</h3>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>fromCurrency</i></b></td>
		        <td><i>String</i></td>
		        <td>Ticker of the currency you want to exchange</td>
		    </tr>
		    <tr>
		        <td><b><i>fromNetwork</i></b></td>
		        <td><i>String</i></td>
		        <td>Network of the currency you want to exchange</td>
		    </tr>
		    <tr>
		        <td><b><i>toCurrency</i></b></td>
		        <td><i>String</i></td>
		        <td>Ticker of the currency you want to receive</td>
		    </tr>
		    <tr>
		        <td><b><i>toNetwork</i></b></td>
		        <td><i>String</i></td>
		        <td>Network of the currency you want to receive</td>
		    </tr>
		    <tr>
		        <td><b><i>flow</i></b></td>
		        <td><i>String</i></td>
		        <td>Type of exchange flow. Enum: ["standard", "fixed-rate"]</td>
		    </tr>
		    <tr>
		        <td><b><i>type</i></b></td>
		        <td><i>String</i></td>
		        <td>Direction of exchange flow. Use "direct" value to set amount for currencyFrom and get amount of currencyTo. Use "reverse" value to set amount for currencyTo and get amount of currencyFrom. Enum: ["direct", "reverse"]</td>
		    </tr>
		    <tr>
		        <td><b><i>rateId</i></b></td>
		        <td><i>String || null</i></td>
		        <td>RateId is needed for fixed-rate flow. If you set param "useRateId" to true, you could use returned field "rateId" in next method for creating transaction to freeze estimated amount that you got in this method. Current estimated amount would be valid until time in field "validUntil"</td>
		    </tr>
		    <tr>
		        <td><b><i>validUntil</i></b></td>
		        <td><i>String || null</i></td>
		        <td>Date and time before estimated amount would be freezed in case of using rateId. If you set param "useRateId" to true, you could use returned field "rateId" in next method for creating transaction to freeze estimated amount that you got in this method. Estimated amount would be valid until this date and time</td>
		    </tr>
		    <tr>
		        <td><b><i>transactionSpeedForecast</i></b></td>
		        <td><i>String || null</i></td>
		        <td>Dash-separated min and max estimated time in minutes</td>
		    </tr>
		    <tr>
		        <td><b><i>warningMessage</i></b></td>
		        <td><i>String || null</i></td>
		        <td>Some warnings like warnings that transactions on this network take longer or that the currency has moved to another network</td>
		    </tr>
		    <tr>
		        <td><b><i>fromAmount</i></b></td>
		        <td><i>Number</i></td>
		        <td>Exchange amount of fromCurrency (in case when type=reverse it is an estimated value)</td>
		    </tr>
		    <tr>
		        <td><b><i>toAmount</i></b></td>
		        <td><i>Number</i></td>
		        <td>Exchange amount of toCurrency (in case when type=direct it is an estimated value)</td>
		    </tr>
		</table>
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    fromamount: (Required if type is direct) Must be greater then 0.
        flow: (Optional) Type of exchange flow. Enum: [\\\"standard\\\", \\\"fixed-rate\\\"]. Default value is standard
        tocurrency: (Required) Ticker of the currency you want to receive
        userateid: (Optional) Use rateId for fixed-rate flow. If this field is true, you could use returned field \\\"rateId\\\" in next method for creating transaction to freeze estimated amount that you got in this method. Current estimated amount would be valid until time in field \\\"validUntil\\\"
        type: (Optional) Direction of exchange flow. Enum: [\\\"direct\\\", \\\"reverse\\\"]. Default value is direct
        fromnetwork: (Optional) Network of the currency you want to exchange
        tonetwork: (Optional) Network of the currency you want to receive
        toamount: (Required if type is reverse) Must be greater then 0.
        fromcurrency: (Required) Ticker of the currency you want to exchange
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/exchange/estimated-amount"
    querystring = {'fromAmount': fromamount, }
    if flow:
        querystring['flow'] = flow
    if tocurrency:
        querystring['toCurrency'] = tocurrency
    if userateid:
        querystring['useRateId'] = userateid
    if type:
        querystring['type'] = type
    if fromnetwork:
        querystring['fromNetwork'] = fromnetwork
    if tonetwork:
        querystring['toNetwork'] = tonetwork
    if toamount:
        querystring['toAmount'] = toamount
    if fromcurrency:
        querystring['fromCurrency'] = fromcurrency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_transaction_status(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns the status and additional information of a single transaction. Transaction ID is taken from the <a href="#3c8edfd0-ae3f-4738-a01c-de3e196bd761">'Create transaction'</a> endpoint.
		
		**Note:** we also give the opportunity to transfer additional fields in the ['Create transaction'](#3c8edfd0-ae3f-4738-a01c-de3e196bd761) endpoint, which we return in this method. Аdditional fields that can be transferred include:
		
		*   userId — user ID;
		*   payload — object that can contain up to 5 arbitrary fields up to 64 characters long;
		    
		
		If you would like to enable these fields, please contact us at [api@changenow.io](mailto:api@changenow.io) with the subject line "Special partner fields".
		
		<h3>Successful response:</h3><p>The response contains an object with transaction information.</p><p>Fields in the response vary depending on the status and a type of the transaction.</p><h5>Successful response fields</h5><table><tbody><tr><td><b>Name</b></td><td><b>Type</b></td><td><b>Description</b></td></tr><tr><td><i><b>id</b></i></td><td><i>String</i></td><td>Transaction ID</td></tr><tr><td><i><b>status</b></i></td><td><i>String</i></td><td>Transaction status:<br>new,<br>waiting,<br>confirming,<br>exchanging,<br>sending,<br>finished,<br>failed,<br>refunded,<br>verifying<br></td></tr><tr><td><i><b>actionsAvailable</b></i></td><td><i>Boolean</i></td><td>Indicates if an exchange can be pushed or refunded using Public push &amp; refund endpoints. <a href="https://documenter.getpostman.com/view/8180765/TzJoFLtG#acf2515b-99c7-44bd-935c-dc42693b8026">Read more</a></td></tr><tr><td><i><b>fromCurrency</b></i></td><td><i>String</i></td><td>Ticker of the currency you want to exchange</td></tr><tr><td><i><b>fromNetwork</b></i></td><td><i>String</i></td><td>Network of the currency you want to exchange</td></tr><tr><td><i><b>toCurrency</b></i></td><td><i>String</i></td><td>Ticker of the currency you want to receive</td></tr><tr><td><i><b>toNetwork</b></i></td><td><i>String</i></td><td>Network of the currency you want to receive</td></tr><tr><td><i><b>expectedAmountFrom</b></i></td><td><i>Number|null</i></td><td>The amount you want to send</td></tr><tr><td><i><b>expectedAmountTo</b></i></td><td><i>Number|null</i></td><td>Estimated value that you will get based on the field <i>expectedAmountFrom</i>.</td></tr><tr><td><i><b>amountFrom</b></i></td><td><i>Number|null</i></td><td>Exchange amount of fromCurrency</td></tr><tr><td><i><b>amountTo</b></i></td><td><i>Number|null</i></td><td>Exchange amount of toCurrency</td></tr><tr><td><i><b>payinAddress</b></i></td><td><i>String</i></td><td>We generate it when creating a transaction</td></tr><tr><td><i><b>payoutAddress</b></i></td><td><i>String</i></td><td>The wallet address that will recieve the exchanged funds</td></tr><tr><td><i><b>payinExtraId</b></i></td><td><i>String|null</i></td><td>We generate it when creating a transaction</td></tr><tr><td><i><b>payoutExtraId</b></i></td><td><i>String|null</i></td><td>Extra ID that you send when creating a transaction</td></tr><tr><td><i><b>refundAddress</b></i></td><td><i>String|null</i></td><td>Refund address (if you specified it)</td></tr><tr><td><i><b>refundExtraId</b></i></td><td><i>String|null</i></td><td>ExtraId for refund (if you specified it)</td></tr><tr><td><i><b>createdAt</b></i></td><td><i>String</i></td><td>Transaction creation date and time</td></tr><tr><td><i><b>updatedAt</b></i></td><td><i>String</i></td><td>Date and time of the last transaction update (e.g. status update)</td></tr><tr><td><i><b>depositReceivedAt</b></i></td><td><i>String|null</i></td><td>Deposit receiving date and time</td></tr><tr><td><i><b>payinHash</b></i></td><td><i>String|null</i></td><td>Transaction hash in the blockchain of the currency which you specified in the fromCurrency field that you send when creating the transaction</td></tr><tr><td><i><b>payoutHash</b></i></td><td><i>String|null</i></td><td>Transaction hash in the blockchain of the currency which you specified in the toCurrency field. We generate it when creating a transaction</td></tr><tr><td><i><b>fromLegacyTicker</b></i></td><td><i>String</i></td><td>Ticker of the currency you want to exchange in an old format as it is specified in a response from <a href="#a44b3f19-3e57-4f39-9822-e2ca3cf1d566">Currency info API-v1 endpoint</a></td></tr><tr><td><i><b>toLegacyTicker</b></i></td><td><i>String</i></td><td>Ticker of the currency you want to receive in an old format as it is specified in a response from <a href="#a44b3f19-3e57-4f39-9822-e2ca3cf1d566">Currency info API-v1 endpoint</a></td></tr></tbody></table><p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p><h3>Request Parameters:</h3>"
    id: (Required) Transaction ID from <a href=\"#3c8edfd0-ae3f-4738-a01c-de3e196bd761\">Create transaction</a> request
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/exchange/by-id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v2_list_of_all_available_pairs(flow: str=None, fromcurrency: str=None, tocurrency: str=None, tonetwork: str=None, fromnetwork: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>This API endpoint returns the list of all available pairs. Some currencies get enabled or disabled from time to time, so make sure to refresh the list occasionally.</p>
		<p>Notice that the resulting array will contain about 13000 pairs.</p>
		
		<h3>Successful response:</h3>
		<p>The response contains an array of underscore separated pair of tickers.</p>
		
		<h3>Request Parameters:</h3>"
    flow: Type of exchange flow. Enum: [\\\"standard\\\", \\\"fixed-rate\\\"]
        fromcurrency: Ticker of the currency you want to exchange
        tocurrency: Ticker of the currency you want to receive
        tonetwork: Network of the currency you want to receive
        fromnetwork: Network of the currency you want to exchange
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v2/exchange/available-pairs"
    querystring = {}
    if flow:
        querystring['flow'] = flow
    if fromcurrency:
        querystring['fromCurrency'] = fromcurrency
    if tocurrency:
        querystring['toCurrency'] = tocurrency
    if tonetwork:
        querystring['toNetwork'] = tonetwork
    if fromnetwork:
        querystring['fromNetwork'] = fromnetwork
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_info(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method returns the name of the coin, the coin's main features (such as anonymity, the need for Extra ID, and logo), and the coin's supported wallets.
		
		<h3>Successful response:</h3>
			<p>The response contains an object with currency information.</p>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>ticker</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency ticker</td>
		    </tr>
		    <tr>
		        <td><b><i>name</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency name</td>
		    </tr>
		    <tr>
		        <td><b><i>image</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency logo url</td>
		    </tr>
		    <tr>
		        <td><b><i>warnings</i></b></td>
		        <td><i>String</i></td>
		        <td>Some warnings like warnings that transactions on this network take longer or that the currency has moved to another network. Field “from” for warnings in case of exchange of this currency and field “to” for warnings in case of exchange for this currency, respectively</td>
		    </tr>
		    <tr>
		        <td><b><i>hasExternalId</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency has an Extra ID</td>
		    </tr>
		    <tr>
		        <td><b><i>isFiat</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is a fiat currency (EUR, USD)</td>
		    </tr>
		    <tr>
		        <td><b><i>isAnonymous</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency has the ability to hide their transactions. This means that even if you know someone’s address, you won’t be able to recognize the balance and receive information about transactions without additional information (e.g. XMR)</td>
		    </tr>
		    <tr>
		        <td><b><i>wallets</i></b></td>
		        <td><i>Object</i></td>
		        <td>This field contains a list of primary and secondary wallets. For each wallet, this endpoint returns the name, url, logo url, supported platforms, degree of anonymity, degree of security, application weight and indicates if the wallet supports different currencies</td>
		    </tr>
		    <tr>
		        <td><b><i>addressExplorerMask</i></b></td>
		        <td><i>String</i></td>
		        <td>This field helps to create a link for the wallet address. Wallet address url = this field.replace('$$', payinAddress or payoutAddress from the <a href="#fa12244b-f879-4675-a6f7-553cc59435dc">Transaction status API endpoint</a>)</td>
		    </tr>
		    <tr>
		        <td><b><i>transactionExplorerMask</i></b></td>
		        <td><i>String</i></td>
		        <td>This field helps to create a link for the transaction hash. Transaction hash url = this field.replace('$$', payinHash or payoutHash from the <a href="#fa12244b-f879-4675-a6f7-553cc59435dc">Transaction status API endpoint</a>) </td>
		    </tr>
		</table>
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    ticker: (Required) Currency ticker
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v1/currencies/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_available_currencies(fixedrate: str='true', active: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns the list of available currencies. Some currencies get enabled or disabled from time to time, so make sure to refresh the list occasionally.
		
		<h3>Successful response:</h3>
			<p>The response contains an array of objects with currency information.</p>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>ticker</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency ticker</td>
		    </tr>
		    <tr>
		        <td><b><i>name</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency name</td>
		    </tr>
		    <tr>
		        <td><b><i>image</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency logo url</td>
		    </tr>
		    <tr>
		        <td><b><i>hasExternalId</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency has an Extra ID</td>
		    </tr>
		    <tr>
		        <td><b><i>isFiat</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is a fiat currency (EUR, USD)</td>
		    </tr>
		    <tr>
		        <td><b><i>featured</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is popular</td>
		    </tr>
		    <tr>
		        <td><b><i>isStable</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is stable</td>
		    </tr>
		    <tr>
		        <td><b><i>supportsFixedRate</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is available on a fixed-rate flow</td>
		    </tr>
		</table>
		
		<h3>Request Parameters:</h3>"
    fixedrate: (Optional) Set true to return only  for the currencies  available on a fixed-rate flow
        active: (Optional) Set true to return only active currencies
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v1/currencies"
    querystring = {}
    if fixedrate:
        querystring['fixedRate'] = fixedrate
    if active:
        querystring['active'] = active
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_available_currencies_for_a_specific_currency(ticker: str, fixedrate: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns the array of markets available for the specified currency be default. The availability of a particular pair is determined by the 'isAvailable' field. Some currencies get enabled or disabled from time to time, so make sure to refresh the list occasionally.
		
		<h3>Successful response:</h3>
			<p>The response contains an array of objects with currencies information.</p>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>ticker</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency ticker</td>
		    </tr>
		    <tr>
		        <td><b><i>name</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency name</td>
		    </tr>
		    <tr>
		        <td><b><i>image</i></b></td>
		        <td><i>String</i></td>
		        <td>Currency logo url</td>
		    </tr>
		    <tr>
		        <td><b><i>hasExternalId</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency has an Extra ID</td>
		    </tr>
		    <tr>
		        <td><b><i>isFiat</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is a fiat currency (EUR, USD)</td>
		    </tr>
		    <tr>
		        <td><b><i>featured</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is popular</td>
		    </tr>
		    <tr>
		        <td><b><i>isStable</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is stable</td>
		    </tr>
		    <tr>
		        <td><b><i>supportsFixedRate</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates if a currency is available on a fixed-rate flow</td>
		    </tr>
		    <tr>
		        <td><b><i>isAvailable</i></b></td>
		        <td><i>Boolean</i></td>
		        <td>Indicates whether the pair is currently supported by our service</td>
		    </tr>
		</table>
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    ticker: (Required) Currency ticker
        fixedrate: (Optional) Set true to return only for the currencies available on a fixed-rate flow
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v1/currencies-to/{ticker}"
    querystring = {}
    if fixedrate:
        querystring['fixedRate'] = fixedrate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minimal_exchange_amount(api_key: str='your_api_key', from_to: str='btc_eth', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API endpoint returns minimal payment amount required to make an exchange. If you try to exchange less, the transaction will most likely fail.
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    from_to: Underscrore separated pair of tickers
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v1/min-amount/{from_to}"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    if from_to:
        querystring['from_to'] = from_to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_range(from_to: str, api_key: str='your_api_key', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<b>New!</b>
		
		The API endpoint returns minimal payment amount and maximum payment amount required to make an exchange. If you try to exchange less than minimum or more than maximum, the transaction will most likely fail. Any pair of assets has minimum amount and some of pairs have maximum amount.
		
		<h3>Successful response:</h3>
		
		<h5>Successful response fields</h5>
		
		<table>
			<tr>
		        <td><b>Name</b></td>
		        <td><b>Type</b></td>
		        <td><b>Description</b></td>
		    </tr>
		    <tr>
		        <td><b><i>minAmount</i></b></td>
		        <td><i>Number</i></td>
		        <td>Minimal payment amount</td>
		    </tr>
		    <tr>
		        <td><b><i>maxAmount</i></b></td>
		        <td><i>Number|null</i></td>
		        <td>Maximum payment amount. Could be null.</td>
		    </tr>
		</table>
		
		<p>You can find <b>examples of errors</b> in the Example request block (use the drop-down list).</p>
		
		<h3>Request Parameters:</h3>"
    from_to: (Required) Underscore separated pair of tickers
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v1/exchange-range/{from_to}"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_available_pairs(includepartners: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>This API endpoint returns the list of all available pairs. Some currencies get enabled or disabled from time to time, so make sure to refresh the list occasionally.</p>
		<p>Notice that the resulting array will contain about 13000 pairs.</p>
		
		<h3>Successful response:</h3>
		<p>The response contains an array of underscore separated pair of tickers.</p>
		
		<h3>Request Parameters:</h3>"
    includepartners: Set false to return all available pairs, except pairs supported by our partners
        
    """
    url = f"https://changenow-crypto-exchange.p.rapidapi.com/v1/market-info/available-pairs/"
    querystring = {}
    if includepartners:
        querystring['includePartners'] = includepartners
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "changenow-crypto-exchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

