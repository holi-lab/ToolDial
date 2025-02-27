import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_status(ref_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<table><thead><tr><th>Parameter</th> <th>Deskripsi</th> <th>Contoh</th></tr></thead> <tbody><tr><td><code>ref_id</code></td> <td>Referensi ID transaksi</td> <td><code>trx1</code> atau <code>41241211133</code></td></tr></tbody></table>"
    
    """
    url = f"https://ppob.p.rapidapi.com/prabayar/check-status/{ref_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppob.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Account is used to display user information according to the Token embedded in the header request."
    
    """
    url = f"https://ppob.p.rapidapi.com/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppob.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deposit_nominal_bank(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Deposit Nominal & Bank is used to display available deposit information on PulsaXpress as well as bank account information used for the deposit process. It also allows users to make a deposit by choosing the desired deposit amount and payment method."
    
    """
    url = f"https://ppob.p.rapidapi.com/payment/deposit"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppob.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indonesia_credit_pricelist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint is for Indonesia only.
		The Credit & Data Pricelist API is used to display the list of available credit and data prices on PulsaXpress. This information can be used to know the price of the products you want to buy before making a transaction. This price information can be updated according to the current prices and can be used to determine the prices offered on your application."
    
    """
    url = f"https://ppob.p.rapidapi.com/prabayar/pulsa/operator"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppob.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def internasional_credit_pricelist(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Pricelist API is used to display all available Pulsa and Data products. The result of this API can obtain the Code, Name, Nominal, and Price of each available product.
		
		<table><thead><tr><th>Country</th> <th>Code</th></tr></thead> <tbody><tr><td>China</td> <td><code>China</code></td></tr> <tr><td>Malaysia</td> <td><code>Malaysia</code></td></tr> <tr><td>Philippines</td> <td><code>Philippines</code></td></tr> <tr><td>Singapore</td> <td><code>Singapore</code></td></tr> <tr><td>Thailand</td> <td><code>Thailand</code></td></tr> <tr><td>Vietnam</td> <td><code>Vietnam</code></td></tr></tbody></table>"
    
    """
    url = f"https://ppob.p.rapidapi.com/prabayar/{code}/operator"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ppob.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

