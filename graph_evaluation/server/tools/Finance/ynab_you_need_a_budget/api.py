import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listtransactions(budget_id: str, last_knowledge_of_server: int, since_date: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns budget transactions"
    last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/transactions"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, 'since_date': since_date, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singletransaction(transaction_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single transaction"
    transaction_id: (Required) The id of the transaction
        budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/transactions/{transaction_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listpayees(budget_id: str, last_knowledge_of_server: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all payees"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/payees"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listlocationsforapayee(payee_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all payee locations for a specified payee"
    payee_id: (Required) id of payee
        budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/payees/{payee_id}/payee_locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singleaccount(budget_id: str, account_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single account"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        account_id: (Required) The id of the account
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/accounts/{account_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlepayee(payee_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single payee"
    payee_id: (Required) The id of the payee
        budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/payees/{payee_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listaccounttransactions(last_knowledge_of_server: int, type: str, since_date: str, account_id: str, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all transactions for a specified account"
    last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        account_id: (Required) The id of the account
        budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/accounts/{account_id}/transactions"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, 'type': type, 'since_date': since_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listpayeetransactions(last_knowledge_of_server: int, since_date: str, budget_id: str, type: str, payee_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all transactions for a specified payee"
    last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        payee_id: (Required) The id of the payee
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/payees/{payee_id}/transactions"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, 'since_date': since_date, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountlist(budget_id: str, last_knowledge_of_server: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all accounts"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/accounts"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlescheduledtransaction(budget_id: str, scheduled_transaction_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single scheduled transaction"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        scheduled_transaction_id: (Required) The id of the scheduled transaction
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlebudget(last_knowledge_of_server: int, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single budget with all related entities.  This resource is effectively a full budget export."
    last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listscheduledtransactions(budget_id: str, last_knowledge_of_server: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scheduled transactions"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/scheduled_transactions"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbudgets(include_accounts: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns budgets list with summary information"
    include_accounts: Whether to include the list of budget accounts
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets"
    querystring = {'include_accounts': include_accounts, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def userinfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns authenticated user information"
    
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/user"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def budgetsettings(budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns settings for a budget"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlebudgetmonth(budget_id: str, month: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single budget month"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        month: (Required) The budget month in ISO format (e.g. 2016-12-01) ("current" can also be used to specify the current calendar month (UTC))
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/months/{month}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlecategoryforaspecificbudgetmonth(budget_id: str, month: str, category_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single category for a specific budget month.  Amounts (budgeted, activity, balance, etc.) are specific to the current budget month (UTC)."
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        month: (Required) The budget month in ISO format (e.g. 2016-12-01) ("current" can also be used to specify the current calendar month (UTC))
        category_id: (Required) The id of the category
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/months/{month}/categories/{category_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcategories(budget_id: str, last_knowledge_of_server: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all categories grouped by category group.  Amounts (budgeted, activity, balance, etc.) are specific to the current budget month (UTC)."
    last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/categories"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbudgetmonths(last_knowledge_of_server: int, budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all budget months"
    last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/months"
    querystring = {'last_knowledge_of_server': last_knowledge_of_server, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcategorytransactions(since_date: str, budget_id: str, category_id: str, last_knowledge_of_server: int, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all transactions for a specified category"
    since_date: If specified, only transactions on or after this date will be included.  The date should be ISO formatted (e.g. 2016-12-30).
        budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        category_id: (Required) The id of the category
        last_knowledge_of_server: The starting server knowledge.  If provided, only entities that have changed since `last_knowledge_of_server` will be included.
        type: If specified, only transactions of the specified type will be included. "uncategorized" and "unapproved" are currently supported.
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/categories/{category_id}/transactions"
    querystring = {'since_date': since_date, 'last_knowledge_of_server': last_knowledge_of_server, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlecategory(budget_id: str, category_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single category.  Amounts (budgeted, activity, balance, etc.) are specific to the current budget month (UTC)."
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        category_id: (Required) The id of the category
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/categories/{category_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def singlepayeelocation(budget_id: str, payee_location_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single payee location"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        payee_location_id: (Required) id of payee location
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/payee_locations/{payee_location_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listpayeelocations(budget_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all payee locations"
    budget_id: (Required) The id of the budget. "last-used" can be used to specify the last used budget and "default" can be used if default budget selection is enabled (see: https://api.youneedabudget.com/#oauth-default-budget).
        
    """
    url = f"https://ynab-you-need-a-budget.p.rapidapi.com/budgets/{budget_id}/payee_locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ynab-you-need-a-budget.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

