import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def account(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of the balances for the client accounts"
    
    """
    url = f"https://finhost.p.rapidapi.com/account/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_details_account(account: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Account bank detailes"
    account: Account ID
        
    """
    url = f"https://finhost.p.rapidapi.com/account/details/{account}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posting_account(account: str, limit: int=10, lastactionkey: str='2022-08-15T12:46:21.379Z', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List assets transfers for the account"
    account: Account identifier
        limit: Response page size
        lastactionkey: Last actionKey to start with the next page
        
    """
    url = f"https://finhost.p.rapidapi.com/posting/{account}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if lastactionkey:
        querystring['lastActionKey'] = lastactionkey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipient_search_currency(value: str, currency: str, attribute: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive recipient info and default account"
    value: Value of the search attribute
        currency: Currency code (ISO 4217)
        attribute: Attribute to use for search
        
    """
    url = f"https://finhost.p.rapidapi.com/recipient/search/{currency}"
    querystring = {'value': value, 'attribute': attribute, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipient(limit: int=10, lastrecipientkey: str='recipient#1660831535', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of the recipients for the client"
    limit: Response page size (default is equal to 10)
        lastrecipientkey: Last recipientKey to start with the next page
        
    """
    url = f"https://finhost.p.rapidapi.com/recipient"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if lastrecipientkey:
        querystring['lastRecipientKey'] = lastrecipientkey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sumsub_access_token(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests customer token to initialize Sumsub SDK"
    
    """
    url = f"https://finhost.p.rapidapi.com/sumsub/access-token"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def client(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Client info"
    
    """
    url = f"https://finhost.p.rapidapi.com/client"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fee_payment(amount: int, account: str, action: str, actiontype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates transfer payment according to the fee plan"
    amount: Client provided amount of assets
        account: Sender account
        action: Transfer type
        actiontype: Transfer type
        
    """
    url = f"https://finhost.p.rapidapi.com/fee/payment"
    querystring = {'amount': amount, 'account': account, 'action': action, 'actionType': actiontype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fee_plans(account: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Describes fee payments for the services"
    account: Sender account
        
    """
    url = f"https://finhost.p.rapidapi.com/fee/plans"
    querystring = {'account': account, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finhost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

