import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettransactionsbycategory(budget_id: str, category_id: str, last_knowledge_of_server: int=None, since_date: str=None, type: str='uncategorized', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all transactions for a specified category"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        category_id: The id of the category
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/categories/{category_id}/transactions"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    if since_date:
        querystring['since_date'] = since_date
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionsbypayee(budget_id: str, payee_id: str, type: str='uncategorized', last_knowledge_of_server: int=None, since_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all transactions for a specified payee"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        payee_id: The id of the payee
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/payees/{payee_id}/transactions"
    querystring = {}
    if type:
        querystring['type'] = type
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    if since_date:
        querystring['since_date'] = since_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionsbyaccount(budget_id: str, account_id: str, type: str='uncategorized', last_knowledge_of_server: int=None, since_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all transactions for a specified account"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        account_id: The id of the account
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/accounts/{account_id}/transactions"
    querystring = {}
    if type:
        querystring['type'] = type
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    if since_date:
        querystring['since_date'] = since_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactions(budget_id: str, last_knowledge_of_server: int=None, since_date: str=None, type: str='uncategorized', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns budget transactions"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/transactions"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    if since_date:
        querystring['since_date'] = since_date
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionbyid(budget_id: str, transaction_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single transaction"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        transaction_id: The id of the transaction
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/transactions/{transaction_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategorybyid(budget_id: str, category_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single category.  Amounts (budgeted, activity, balance, etc.) are specific to the current budget month (UTC)."
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        category_id: The id of the category
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/categories/{category_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpayeelocationsbypayee(payee_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all payee locations for a specified payee"
    payee_id: id of payee
        budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/payees/{payee_id}/payee_locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudgetbyid(budget_id: str, last_knowledge_of_server: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single budget with all related entities.  This resource is effectively a full budget export."
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmonthcategorybyid(month: str, category_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single category for a specific budget month.  Amounts (budgeted, activity, balance, etc.) are specific to the current budget month (UTC)."
    month: The budget month in ISO format (e.g. 2016-12-01) ("current" can also be used to specify the current calendar month (UTC))
        category_id: The id of the category
        budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/months/{month}/categories/{category_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategories(budget_id: str, last_knowledge_of_server: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all categories grouped by category group.  Amounts (budgeted, activity, balance, etc.) are specific to the current budget month (UTC)."
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/categories"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpayeelocationbyid(payee_location_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single payee location"
    payee_location_id: id of payee location
        budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/payee_locations/{payee_location_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudgetsettingsbyid(budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns settings for a budget"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudgets(include_accounts: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns budgets list with summary information"
    include_accounts: Whether to include the list of budget accounts
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets"
    querystring = {}
    if include_accounts:
        querystring['include_accounts'] = include_accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpayeelocations(budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all payee locations"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/payee_locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudgetmonths(budget_id: str, last_knowledge_of_server: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all budget months"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/months"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudgetmonth(budget_id: str, month: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single budget month"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        month: The budget month in ISO format (e.g. 2016-12-01) ("current" can also be used to specify the current calendar month (UTC))
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/months/{month}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpayeebyid(budget_id: str, payee_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single payee"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        payee_id: The id of the payee
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/payees/{payee_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuser(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns authenticated user information"
    
    """
    url = f"https://ynab16.p.rapidapi.com/user"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccountbyid(account_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single account"
    account_id: The id of the account
        budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/accounts/{account_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccounts(budget_id: str, last_knowledge_of_server: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all accounts"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/accounts"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpayees(budget_id: str, last_knowledge_of_server: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all payees"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/payees"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getscheduledtransactionbyid(scheduled_transaction_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single scheduled transaction"
    scheduled_transaction_id: The id of the scheduled transaction
        budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getscheduledtransactions(budget_id: str, last_knowledge_of_server: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scheduled transactions"
    budget_id: The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab16.p.rapidapi.com/budgets/{budget_id}/scheduled_transactions"
    querystring = {}
    if last_knowledge_of_server:
        querystring['last_knowledge_of_server'] = last_knowledge_of_server
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

