import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def balance(currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    currency: Currency to convert to.

        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/aggregate/balance"
    querystring = {}
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrentexchangerate(pairs: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    pairs: Currency pairs for which exchange rate should be returned
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/exchangerate/current"
    querystring = {'pairs': pairs, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authuser(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: User id
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/users/{is_id}/auth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethistoricalexchangerate(days: str, pair: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    days: Historical dates for which exchange rates should be returned
        pair: Currency pair for which exchange rates should be returned
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/exchangerate/history"
    querystring = {'days': days, 'pair': pair, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountproviderslist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/aggregate/account_providers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuser(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: User id
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/users/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountproviderhintslist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/aggregate/account_provider_hints"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccounthint(account_provider: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    account_provider: URL value from account_providers method
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/{account_provider}/connect"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userapi_authenticateproject(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/auth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionriskscoredetails(transaction: str='0638e15482e9d0fdb08920666390a546f32dd6ab15ffc81ed97e67b73b0d7205', chain: str='btc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    transaction: Transaction hash for which risk should be returned
        chain: Blockchain identifier
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/risk/transaction/score/details"
    querystring = {}
    if transaction:
        querystring['transaction'] = transaction
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accounts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/aggregate/accounts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transactions(currency: str='USD', since: str='2020-01-01', account_filter: str='497f6eca-6276-4993-bfeb-53cbbbba6f08', until: str='2020-02-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    currency: Currency to convert to.

        since: Set time from which the transactions will be get.
The parameter is passed as-is to backend services.
The default value is 30 days before the actual date or 30 days before the date specified in "until" parameter.

        account_filter: Filter results to only provided account.
When omitted, it returns all transactions from all accounts.

        until: Set time to which the transactions will be get.
The parameter is passed as-is to backend services.
The default value is the actual date.

        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/aggregate/transactions"
    querystring = {}
    if currency:
        querystring['currency'] = currency
    if since:
        querystring['since'] = since
    if account_filter:
        querystring['account-filter'] = account_filter
    if until:
        querystring['until'] = until
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaddressnameinfo(address: str, chain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    address: Address for wich name and category should be returned
        chain: Blockchain identifier
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/addressname/simple"
    querystring = {'address': address, 'chain': chain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionriskscorecase(case_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    case_id: Case identifier
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/risk/transaction/score/details/{case_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userapi_authenticatedeveloper(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/auth/developer"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaddressriskscorecase(case_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    case_id: Case identifier
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/risk/score/details/{case_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaddressriskscoredetails(address: str='bc1qjl7k0dpcsw3djmzq25qv6peavgxysq95pcduuq', chain: str='btc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    address: Address for which risk should be returned
        chain: Blockchain identifier
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/risk/score/details"
    querystring = {}
    if address:
        querystring['address'] = address
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaddressriskscore(address: str='bc1qjl7k0dpcsw3djmzq25qv6peavgxysq95pcduuq', chain: str='btc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    address: Address for which risk should be returned
        chain: Blockchain identifier
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/risk/score"
    querystring = {}
    if address:
        querystring['address'] = address
    if chain:
        querystring['chain'] = chain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionriskscore(chain: str='btc', transaction: str='0638e15482e9d0fdb08920666390a546f32dd6ab15ffc81ed97e67b73b0d7205', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    chain: Blockchain identifier
        transaction: Transaction hash for which risk should be returned
        
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/risk/transaction/score"
    querystring = {}
    if chain:
        querystring['chain'] = chain
    if transaction:
        querystring['transaction'] = transaction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listusers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://blockmate-crypto-account-connector.p.rapidapi.com/v1/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockmate-crypto-account-connector.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

