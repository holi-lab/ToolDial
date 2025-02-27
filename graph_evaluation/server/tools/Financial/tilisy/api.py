import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_aspsps_aspsps_get(country: str=None, psu_type: str='business', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of ASPSPs with their meta information"
    country: Display only ASPSPs from specified country
        psu_type: Display only ASPSPs which support specified psu type
        
    """
    url = f"https://tilisy.p.rapidapi.com/aspsps"
    querystring = {}
    if country:
        querystring['country'] = country
    if psu_type:
        querystring['psu_type'] = psu_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tilisy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_account_transactions_accounts_account_id_transactions_get(account_id: str, date_from: str=None, continuation_key: str=None, date_to: str=None, transaction_status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of account transactions according to provided parameters"
    account_id: PSU account ID accessible in the provided session
        date_from: Date to fetch transactions from (including the date, UTC timezone is assumed)
        continuation_key: Key, allowing iterate over multiple API pages of transactions
        date_to: Date to fetch transactions to (including the date, UTC timezone is assumed)
        transaction_status: Filter transactions by provided status
        
    """
    url = f"https://tilisy.p.rapidapi.com/accounts/{account_id}/transactions"
    querystring = {}
    if date_from:
        querystring['date_from'] = date_from
    if continuation_key:
        querystring['continuation_key'] = continuation_key
    if date_to:
        querystring['date_to'] = date_to
    if transaction_status:
        querystring['transaction_status'] = transaction_status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tilisy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_application_application_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get application associated with provided JWT key id"
    
    """
    url = f"https://tilisy.p.rapidapi.com/application"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tilisy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_session_data(session_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get session data by session id"
    session_id: Previously authorized session id
        
    """
    url = f"https://tilisy.p.rapidapi.com/sessions/{session_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tilisy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_account_balances(account_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available account balances by provided account id"
    account_id: PSU account ID accessible in the provided session
        
    """
    url = f"https://tilisy.p.rapidapi.com/accounts/{account_id}/balances"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tilisy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

