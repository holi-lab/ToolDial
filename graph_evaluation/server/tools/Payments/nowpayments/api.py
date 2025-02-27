import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_7_getlistofpayments(x_api_key: str, datefrom: str, page: int, limit: int, sortby: str, orderby: str, dateto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the entire list of all transactions, created with certain API key.
		The list of optional parameters:
		- limit - number of records in one page. (possible values: from 1 to 500)
		- page - the page number you want to get (possible values: from 0 to **page count - 1**)
		- sortBy - sort the received list by a paramenter. Set to **created_at** by default (possible values: payment_id, payment_status, pay_address, price_amount, price_currency, pay_amount, actually_paid, pay_currency, order_id, order_description, purchase_id, outcome_amount, outcome_currency)
		- orderBy - display the list in ascending or descending order. Set to **asc** by default (possible values: asc, desc)
		- dateFrom - select the displayed period start date (date format: YYYY-MM-DD or yy-MM-ddTHH:mm:ss.SSSZ).
		- dateTo - select the displayed period end date (date format: YYYY-MM-DD or yy-MM-ddTHH:mm:ss.SSSZ)."
    
    """
    url = f"https://nowpayments.p.rapidapi.com/v1/payment/"
    querystring = {'x-api-key': x_api_key, 'dateFrom': datefrom, 'page': page, 'limit': limit, 'sortBy': sortby, 'orderBy': orderby, 'dateTo': dateto, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nowpayments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_getapistatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a method to get information about the current state of the API. If everything is OK, you will receive an "OK" message. Otherwise, you'll see some error."
    
    """
    url = f"https://nowpayments.p.rapidapi.com/v1/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nowpayments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_getavailablecurrencies(x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a method for obtaining information about the cryptocurrencies available for payments. This depends on the cryptocurrency you choose for your particular store.
		Optional parameters:
		- fixed_rate(optional) - boolean, can be **true** or **false**. Returns avaliable currencies with minimum and maximum amount of the exchange."
    
    """
    url = f"https://nowpayments.p.rapidapi.com/v1/currencies"
    querystring = {'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nowpayments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_getestimatedprice(x_api_key: str, currency_to: str, currency_from: str, amount: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a method for calculating the approximate price in cryptocurrency for a given value in Fiat currency. You will need to provide the initial cost in the Fiat currency (amount, currency_from) and the necessary cryptocurrency (currency_to)
		Currently following fiat currencies are available: usd, eur, nzd, brl, gbp."
    
    """
    url = f"https://nowpayments.p.rapidapi.com/v1/estimate"
    querystring = {'x-api-key': x_api_key, 'currency_to': currency_to, 'currency_from': currency_from, 'amount': amount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nowpayments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_gettheminimumpaymentamount(x_api_key: str, currency_from: str, currency_to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the minimum payment amount for a specific pair.
		
		You can provide both currencies in the pair or just currency_from, and we will calculate the minimum payment amount for currency_from and currency which you have specified as the outcome in the Store Settings.
		
		In the case of several outcome wallets we will calculate the minimum amount in the same way we route your payment to a specific wallet."
    
    """
    url = f"https://nowpayments.p.rapidapi.com/v1/min-amount"
    querystring = {'x-api-key': x_api_key, 'currency_from': currency_from, 'currency_to': currency_to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nowpayments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_getpaymentstatus(x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the actual information about the payment. You need to provide the ID of the payment in the request.
		
		NOTE! You should make the get payment status request with the same API key that you used in the create payment request.
		Here is the list of avalable statuses:
		- waiting - waiting for the customer to send the payment. The initial status of each payment.
		- confirming - the transaction is being processed on the blockchain. Appears when NOWPayments detect the funds from the user on the blockchain.
		- confirmed -  the process is confirmed by the blockchain. Customer’s funds have accumulated enough confirmations.
		- sending - the funds are being sent to your personal wallet. We are in the process of sending the funds to you.
		- partially_paid -  it shows that the customer sent the less than the actual price. Appears when the funds have arrived in your wallet.
		- finished - the funds have reached your personal address and the payment is finished.
		- failed -  the payment wasn't completed due to the error of some kind.
		- refunded -  the funds were refunded back to the user.
		- expired - the user didn't send the funds to the specified address in the 24 hour time window."
    
    """
    url = f"https://nowpayments.p.rapidapi.com/v1/payment/<your_payment_id>"
    querystring = {'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nowpayments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

