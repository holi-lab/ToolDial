import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_token_forwarding_transaction_webhook(transactionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get token forwarding transaction webhook"
    
    """
    url = f"https://token-forwarding.p.rapidapi.com/token-forwarding-transaction/{transactionid}/webhook"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token-forwarding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_forwarding_transactions(updatedlt: str='2022-01-11 06:08:17', updatedgte: str='2023-01-10 06:08:17', is_id: str='dfe02338-43ae-453f-bd51-6f0ea98e4df9', offset: int=0, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get token forwarding transactions"
    
    """
    url = f"https://token-forwarding.p.rapidapi.com/token-forwarding-transaction"
    querystring = {}
    if updatedlt:
        querystring['updatedLt'] = updatedlt
    if updatedgte:
        querystring['updatedGte'] = updatedgte
    if is_id:
        querystring['id'] = is_id
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token-forwarding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_usage_quota_for_the_current_month(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get usage quota for the current month"
    
    """
    url = f"https://token-forwarding.p.rapidapi.com/rapidapi-usage-quota"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token-forwarding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_gas_fee_funding_wallets(is_id: str='dfe02338-43ae-453f-bd51-6f0ea98e4df9', limit: int=100, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get gas fee funding wallets"
    
    """
    url = f"https://token-forwarding.p.rapidapi.com/gas-fee-funding-wallet"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token-forwarding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token_forwarding_wallets(is_id: str='dfe02338-43ae-453f-bd51-6f0ea98e4df9', limit: int=100, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get token forwarding wallets"
    
    """
    url = f"https://token-forwarding.p.rapidapi.com/token-forwarding-wallet"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token-forwarding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

