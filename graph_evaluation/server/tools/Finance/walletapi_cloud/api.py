import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_transactions(is_from: int, to: str, projectid: str, pagesize: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a pageable list of transactions for a time period."
    from: Date from
        to: Date to
        projectid: Project id
        pagesize: Max: 500
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/transactions"
    querystring = {'from': is_from, 'to': to, 'pageSize': pagesize, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_projects(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of your projects."
    
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_project_s_assets(projectid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of assets for a project."
    projectid: Project ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/assets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transaction_details(projectid: str, txid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call this endpoint to obtain a transaction's current state."
    projectid: Project ID
        txid: Transaction ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/transactions/{txid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_wallet_details(projectid: str, walletid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain wallet balance and details."
    projectid: Project ID
        walletid: Wallet ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/wallets/{walletid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_asset_details(assetid: str, projectid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain the details of an asset with provided ID."
    assetid: Asset ID
        projectid: Project ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/currencies/{assetid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_project_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a project's details including all assets."
    id: Project ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_customers(pagesize: int, page: int, projectid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of your customers with pagination. Specify current page and page size as query parameters."
    pagesize: Maximum: 500
        projectid: Project id
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/customers"
    querystring = {'pageSize': pagesize, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_system_wallets(pid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of system wallets for this project."
    pid: Project ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{pid}/system-wallets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_customer_s_wallets(projectid: str, customerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of wallets owned by a customer."
    projectid: Project ID
        customerid: Customer ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/customers/{customerid}/wallets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_customer_details(customerid: str, projectid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call this endpoint to obtain a record for a customer with provided ID."
    customerid: Customer ID
        projectid: Project ID
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/customers/{customerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_operations_in_wallet_s_ledger(projectid: str, pagesize: int, walletid: str, is_from: int, to: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a pageable list of operations that were posted to a wallet within a specified time period."
    projectid: Project ID
        pagesize: Page size, max: 500
        walletid: Wallet ID
        from: Date from
        to: Date to
        page: Page number
        
    """
    url = f"https://walletapi-cloud.p.rapidapi.com/projects/{projectid}/wallets/{walletid}/entries"
    querystring = {'pageSize': pagesize, 'from': is_from, 'to': to, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walletapi-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

