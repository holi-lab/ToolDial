import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getaccountsac(query: str='query-string', limit: int=10, date: str='2020-09-17', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: The autocomplete search query for accounts.
        limit: The number of items returned.
        date: If the account is an asset account or a liability, the autocomplete will also return the balance of the account on this date.
        type: Optional filter on the account type(s) used in the autocomplete.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/accounts"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    if date:
        querystring['date'] = date
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listrulebygroup(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List rules in this rule group."
    id: The ID of the rule group.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/rule_groups/{is_id}/rules"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpenseasset(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, grouped by asset account.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/asset"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listrulebybill(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will list all rules that have an action to set the bill to this bill."
    id: The ID of the bill.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/bills/{is_id}/rules"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionsac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/transactions"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhook(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all info of a single webhook."
    id: The webhook ID.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/webhooks/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbybudget(is_id: str, type: str='all', start: str='2018-09-17', end: str='2018-12-31', limit: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all transactions linked to a budget, possibly limited by start and end"
    id: The ID of the budget.
        type: Optional filter on the transaction type(s) returned
        start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        limit: Limits the number of results on one page.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets/{is_id}/transactions"
    querystring = {}
    if type:
        querystring['type'] = type
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsinglewebhookmessage(is_id: str, messageid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When a webhook is triggered it will store the actual content of the webhook in a webhook message. You can view and analyse a single one using this endpoint."
    id: The webhook ID.
        messageid: The webhook message ID.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/webhooks/{is_id}/messages/{messageid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensenobill(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, including only expenses with no bill.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/no-bill"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listwebhook(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the user's webhooks."
    page: The page number, if necessary. The default pagination is 50, so 50 webhooks per page.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/webhooks"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrenciescodeac(query: str='str', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: The autocomplete search query.
        limit: The number of items returned.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/currencies-with-code"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachmentbytag(tag: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all attachments."
    tag: Either the tag itself or the tag ID.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/tags/{tag}/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbybudgetlimit(limitid: str, is_id: str, type: str='all', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the transactions within one budget limit. The start and end date are dictated by the budget limit."
    limitid: The ID of the budget limit. The budget limit MUST be associated to the budget ID.
        id: The ID of the budget. The budget limit MUST be associated to the budget ID.
        type: Optional filter on the transaction type(s) returned
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets/{is_id}/limits/{limitid}/transactions"
    querystring = {}
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrulesac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/rules"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpiggybank(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single piggy bank."
    id: The ID of the piggy bank.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/piggy_banks/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachmentbybill(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will list all attachments linked to the bill."
    id: The ID of the bill.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/bills/{is_id}/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getobjectgroup(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single object group."
    id: The ID of the object group.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/object_groups/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listpiggybankbyobjectgroup(is_id: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of all the piggy banks connected to the object group.
		"
    id: The ID of the account.
        page: Page number. The default pagination is per 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/object_groups/{is_id}/piggy_banks"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listobjectgroups(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all oject groups."
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/object_groups"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudgetlimit(limitid: int, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limitid: The ID of the budget limit. The budget limit MUST be associated to the budget ID.
        id: The ID of the budget. The budget limit MUST be associated to the budget ID.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets/{is_id}/limits/{limitid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listpiggybank(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all piggy banks."
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/piggy_banks"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensebudget(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', budgets: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, grouped by (any) budget.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        budgets: The budgets to be included in the results.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/budget"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    if budgets:
        querystring['budgets[]'] = budgets
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listeventbypiggybank(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all events linked to a piggy bank (adding and removing money)."
    id: The ID of the piggy bank
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/piggy_banks/{is_id}/events"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensetag(end: str, start: str, tags: str='[
  1,
  2,
  3
]', accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, grouped by (any) tag.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        tags: The tags to be included in the results.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/tag"
    querystring = {'end': end, 'start': start, }
    if tags:
        querystring['tags[]'] = tags
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbill(end: str='2018-12-31', page: int=1, start: str='2018-09-17', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will list all the user's bills."
    end: A date formatted YYYY-MM-DD. If it is added to the request, Firefly III will calculate the appropriate payment and paid dates.

        page: Page number. The default pagination is 50.
        start: A date formatted YYYY-MM-DD. If it is are added to the request, Firefly III will calculate the appropriate payment and paid dates.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/bills"
    querystring = {}
    if end:
        querystring['end'] = end
    if page:
        querystring['page'] = page
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insighttransfers(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the transfers made by the user, grouped by asset account or lability.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only transfers between those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/transfer/asset"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudget(is_id: str, start: str='2018-09-17', end: str='2018-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single budget. If the start date and end date are submitted as well, the "spent" array will be updated accordingly."
    id: The ID of the requested budget.
        start: A date formatted YYYY-MM-DD, to get info on how much the user has spent.

        end: A date formatted YYYY-MM-DD, to get info on how much the user has spent.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets/{is_id}"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensecategory(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', categories: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, grouped by (any) category.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        categories: The categories to be included in the results.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/category"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    if categories:
        querystring['categories[]'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtag(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all of the user's tags."
    page: Page number
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/tags"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbillsac(limit: int=10, query: str='query-string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query for bills.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/bills"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensenocategory(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, including only expenses with no category.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/no-category"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightincometotal(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a sum of the total income received by the user.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only deposits to those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/income/total"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecurringac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/recurring"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachmentbybudget(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all attachments."
    id: The ID of the budget.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets/{is_id}/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getavailablebudget(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single available budget, by ID."
    id: The ID of the available budget.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/available_budgets/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbylinktype(is_id: str, page: int=1, start: str='2018-09-17', type: str='all', end: str='2018-09-17', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all transactions under this link type, both the inward and outward transactions.
		"
    id: The ID of the link type.
        page: Page number. The default pagination is per 50 items.
        start: A date formatted YYYY-MM-DD, to limit the results.

        type: Optional filter on the transaction type(s) returned.
        end: A date formatted YYYY-MM-DD, to limit the results.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/link_types/{is_id}/transactions"
    querystring = {}
    if page:
        querystring['page'] = page
    if start:
        querystring['start'] = start
    if type:
        querystring['type'] = type
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbudgetlimit(end: str, start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all budget limits for for this date range.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budget-limits"
    querystring = {'end': end, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionlink(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single link by its ID.
		"
    id: The ID of the transaction link.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transaction_links/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbudget(start: str='2018-09-17', page: int=1, end: str='2018-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the budgets the user has made. If the start date and end date are submitted as well, the "spent" array will be updated accordingly."
    start: A date formatted YYYY-MM-DD, to get info on how much the user has spent. You must submit both start and end.

        page: Page number. The default pagination is 50.
        end: A date formatted YYYY-MM-DD, to get info on how much the user has spent. You must submit both start and end.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets"
    querystring = {}
    if start:
        querystring['start'] = start
    if page:
        querystring['page'] = page
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpiggiesbalanceac(query: str='str', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: The autocomplete search query.
        limit: The number of items returned.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/piggy-banks-with-balance"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionwithoutbudget(page: int=1, limit: int=5, end: str='2018-12-31', start: str='2018-09-17', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all transactions without a budget, possibly limited by start and end"
    page: Page number. The default pagination is 50.
        limit: Limits the number of results on one page.
        end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        type: Optional filter on the transaction type(s) returned
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets/transactions-without-budget"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlinktype(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single link type by its ID.
		"
    id: The ID of the link type.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/link_types/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrenciesac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/currencies"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettag(tag: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single tag."
    tag: Either the tag itself or the tag ID. If you use the tag itself, and it contains international (non-ASCII) characters, your milage may vary.
        page: Page number
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/tags/{tag}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbudgetsac(query: str='str', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: The autocomplete search query.
        limit: The number of items returned
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/budgets"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listlinktype(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the link types the system has. These include the default ones as well as any new ones.
		"
    page: Page number. The default pagination is 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/link_types"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionlink(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the transaction links.
		"
    page: Page number. The default pagination is per 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transaction_links"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettagac(query: str='str', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: The autocomplete search query.
        limit: The number of items returned.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/tags"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpiggiesac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/piggy-banks"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrentuser(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the currently authenticated user.
		"
    
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/about/user"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcron(clitoken: str, date: str='2018-09-17', force: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Firefly III has one endpoint for its various cron related tasks. Send a GET to this endpoint
		to run the cron. The cron requires the CLI token to be present. The cron job will fire for all
		users.
		"
    clitoken: The CLI token of any user in Firefly III, required to run the cron job.
        date: A date formatted YYYY-MM-DD. This can be used to make the cron job pretend it's running
on another day.

        force: Forces the cron job to fire, regardless of whether it has fired before. This may result
in double transactions or weird budgets, so be careful.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/cron/{clitoken}"
    querystring = {}
    if date:
        querystring['date'] = date
    if force:
        querystring['force'] = force
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachment(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all attachments.
		"
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getabout(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns general system information and versions of the (supporting) software.
		"
    
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/about"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadattachment(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to download the binary content of a transaction. It will be sent to you as a download, using the content type "application/octet-stream" and content disposition "attachment; filename=example.pdf".
		"
    id: The ID of the attachment.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/attachments/{is_id}/download"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getattachment(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single attachment. This endpoint only returns the available metadata for the attachment. Actual file data is handled in two other endpoints (see below).
		"
    id: The ID of the attachment.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/attachments/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconfiguration(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all editable and not-editable configuration values for this Firefly III installation"
    
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/configuration"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listpreference(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all of the preferences of the user."
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/preferences"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecurrence(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single recurring transaction."
    id: The ID of the recurring transaction.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/recurrences/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachmentbytransaction(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all attachments."
    id: The ID of the transaction.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transactions/{is_id}/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listlinksbyjournal(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all the transaction links for an individual journal (a split). Don't use the group ID, you need the actual underlying journal (the split)."
    id: The ID of the transaction journal / the split.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transaction-journals/{is_id}/links"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listeventbytransaction(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all piggy bank events."
    id: The ID of the transaction.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transactions/{is_id}/piggy_bank_events"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrency(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single currency."
    code: The currency code.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbudgetlimitbycurrency(code: str, end: str='2018-01-31', start: str='2018-01-01', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all budget limits with this currency"
    code: The currency code.
        end: End date for the budget limit list.
        start: Start date for the budget limit list.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}/budget_limits"
    querystring = {}
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def testrulegroup(is_id: str, triggered_limit: int=None, end: str='2018-09-17', start: str='2018-09-17', search_limit: int=None, accounts: str='[
  1,
  2,
  3
]', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test which transactions would be hit by the rule group. No changes will be made. Limit the result if you want to."
    id: The ID of the rule group.
        triggered_limit: Maximum number of transactions the rule group can actually trigger on, before Firefly III stops. I would suggest setting this to 10 or 15. Don't go above the user's page size, because browsing to page 2 or 3 of a test result would fire the test again, making any navigation efforts very slow.

        end: A date formatted YYYY-MM-DD, to limit the transactions the test will be applied to. Both the start date and the end date must be present.

        start: A date formatted YYYY-MM-DD, to limit the transactions the test will be applied to. Both the start date and the end date must be present.

        search_limit: Maximum number of transactions Firefly III will try. Don't set this too high, or it will take Firefly III very long to run the test. I suggest a max of 200.

        accounts: Limit the testing of the rule group to these asset accounts or liabilities. Only asset accounts and liabilities will be accepted. Other types will be silently dropped.

        page: Page number. The default pagination is 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/rule_groups/{is_id}/test"
    querystring = {}
    if triggered_limit:
        querystring['triggered_limit'] = triggered_limit
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    if search_limit:
        querystring['search_limit'] = search_limit
    if accounts:
        querystring['accounts[]'] = accounts
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransaction(start: str='2018-09-17', page: int=1, end: str='2018-09-17', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the user's transactions."
    start: A date formatted YYYY-MM-DD. This is the start date of the selected range (inclusive).

        page: Page number. The default pagination is 50.
        end: A date formatted YYYY-MM-DD. This is the end date of the selected range (inclusive).

        type: Optional filter on the transaction type(s) returned.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transactions"
    querystring = {}
    if start:
        querystring['start'] = start
    if page:
        querystring['page'] = page
    if end:
        querystring['end'] = end
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionbyjournal(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single transaction by underlying journal (split)."
    id: The ID of the transaction journal (split).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transaction-journals/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listpiggybankbyaccount(is_id: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of all the piggy banks connected to the account.
		"
    id: The ID of the account.
        page: Page number. The default pagination is per 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/accounts/{is_id}/piggy_banks"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportrules(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your rules and rule groups from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/rules"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransaction(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single transaction."
    id: The ID of the transaction.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/transactions/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listrulebycurrency(code: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all rules with this currency."
    code: The currency code.
        page: Page number. The default pagination per 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}/rules"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listaccountbycurrency(code: str, date: str=None, type: str='all', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all accounts with this currency."
    code: The currency code.
        date: A date formatted YYYY-MM-DD. When added to the request, Firefly III will show the account's balance on that day.

        type: Optional filter on the account type(s) returned
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}/accounts"
    querystring = {}
    if date:
        querystring['date'] = date
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachmentbyaccount(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all attachments."
    id: The ID of the account.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/accounts/{is_id}/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getaccount(is_id: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single account by its ID.
		"
    id: The ID of the account.
        date: A date formatted YYYY-MM-DD. When added to the request, Firefly III will show the account's balance on that day.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/accounts/{is_id}"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listrulegroup(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all rule groups."
    page: Page number. The default pagination is 50
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/rule_groups"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrulegroup(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single rule group. This does not include the rules. For that, see below."
    id: The ID of the rule group.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/rule_groups/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhookmessages(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When a webhook is triggered the actual message that will be send is stored in a "message". You can view and analyse these messages."
    id: The webhook ID.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/webhooks/{is_id}/messages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdefaultcurrency(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's default currency."
    
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/default"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listavailablebudgetbycurrency(code: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available budgets with this currency."
    code: The currency code.
        page: Page number. The default pagination is 50
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}/available_budgets"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportbills(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your bills from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/bills"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactiontypesac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/transaction-types"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insighttransfernotag(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the transfers made by the user, including only transfers with no tag.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only transfers from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/transfer/no-tag"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchtransactions(query: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches through the users transactions."
    query: The query you wish to search for.
        page: Page number. The default pagination is 50
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/search/transactions"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getobjectgroupsac(query: str='str', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: The autocomplete search query.
        limit: The number of items returned.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/object-groups"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbytag(tag: str, type: str='all', page: int=1, end: str='2018-09-17', start: str='2018-09-17', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all transactions with this tag."
    tag: Either the tag itself or the tag ID.
        type: Optional filter on the transaction type(s) returned.
        page: Page number. The default pagination is 50.
        end: A date formatted YYYY-MM-DD. This is the end date of the selected range (inclusive).

        start: A date formatted YYYY-MM-DD. This is the start date of the selected range (inclusive).

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/tags/{tag}/transactions"
    querystring = {}
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightincomerevenue(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the income received by the user, grouped by revenue account.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you add the accounts ID's of revenue accounts, only those accounts
are included in the results. If you include ID's of asset accounts or liabilities, only deposits to those
asset accounts / liabilities will be included. You can combine both asset / liability and deposit account ID's.
Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/income/revenue"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listrecurrencebycurrency(code: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all recurring transactions with this currency."
    code: The currency code.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}/recurrences"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbillbycurrency(code: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all bills with this currency."
    code: The currency code.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}/bills"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbycurrency(code: str, type: str='all', end: str='2018-12-31', page: int=1, start: str='2018-09-17', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all transactions with this currency."
    code: The currency code.
        type: Optional filter on the transaction type(s) returned
        end: A date formatted YYYY-MM-DD, to limit the list of transactions.

        page: Page number. The default pagination is per 50.
        start: A date formatted YYYY-MM-DD, to limit the list of transactions.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies/{code}/transactions"
    querystring = {}
    if type:
        querystring['type'] = type
    if end:
        querystring['end'] = end
    if page:
        querystring['page'] = page
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensenotag(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, including only expenses with no tag.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/no-tag"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcurrency(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all currencies."
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/currencies"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightincomeasset(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the income received by the user, grouped by asset account.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only deposits to those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/income/asset"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insighttransfertag(start: str, end: str, tags: str='[
  1,
  2,
  3
]', accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the transfers created by the user, grouped by (any) tag.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        tags: The tags to be included in the results.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only transfers between those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/transfer/tag"
    querystring = {'start': start, 'end': end, }
    if tags:
        querystring['tags[]'] = tags
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbyaccount(is_id: str, start: str='2018-09-17', end: str='2018-09-17', type: str='all', limit: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of all the transactions connected to the account.
		"
    id: The ID of the account.
        start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        type: Optional filter on the transaction type(s) returned.
        limit: Limits the number of results on one page.
        page: Page number. The default pagination is per 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/accounts/{is_id}/transactions"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportcategories(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your categories from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/categories"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportpiggies(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your piggy banks from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/piggy-banks"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensenobudget(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, including only expenses with no budget.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/no-budget"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategoriesac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/categories"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactionsidac(query: str='str', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    query: The autocomplete search query.
        limit: The number of items returned.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/transactions-with-id"
    querystring = {}
    if query:
        querystring['query'] = query
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhookmessageattempts(messageid: int, is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When a webhook message fails to send it will store the failure in an "attempt". You can view and analyse these. Webhook messages that receive too many attempts (failures) will not be sent again. You must first clear out old attempts before the message can go out again."
    messageid: The webhook message ID.
        id: The webhook ID.
        page: Page number. The default pagination is per 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/webhooks/{is_id}/messages/{messageid}/attempts"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportbudgets(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your budgets and associated budget data from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/budgets"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpenseexpense(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, grouped by expense account.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you add the accounts ID's of expense accounts, only those accounts
are included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. You can combine both asset / liability and expense account ID's.
Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/expense"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exporttransactions(start: str, end: str, accounts: str='1,2,3', type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export transactions from Firefly III into a file. Currently supports CSV exports only.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: Limit the export of transactions to these accounts only. Only asset accounts will be accepted. Other types will be silently dropped.

        type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/transactions"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts'] = accounts
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insighttransfernocategory(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the transfers made by the user, including only transfers with no category.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only transfers between those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/transfer/no-category"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insighttransfercategory(end: str, start: str, categories: str='[
  1,
  2,
  3
]', accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the transfers made by the user, grouped by (any) category.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        categories: The categories to be included in the results.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only transfers between those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/transfer/category"
    querystring = {'end': end, 'start': start, }
    if categories:
        querystring['categories[]'] = categories
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportaccounts(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your accounts from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/accounts"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exporttags(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your tags from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/tags"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachmentbypiggybank(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all attachments."
    id: The ID of the piggy bank.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/piggy_banks/{is_id}/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchaccounts(query: str, field: str, type: str='all', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for accounts"
    query: The query you wish to search for.
        field: The account field(s) you want to search in.
        type: The type of accounts you wish to limit the search to.
        page: Page number. The default pagination is 50
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/search/accounts"
    querystring = {'query': query, 'field': field, }
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightincomenocategory(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the income received by the user, including only income with no category.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only deposits to those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/income/no-category"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listaccount(type: str='all', date: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of all the accounts owned by the authenticated user.
		"
    type: Optional filter on the account type(s) returned
        date: A date formatted YYYY-MM-DD. When added to the request, Firefly III will show the account's balance on that day.

        page: Page number. The default pagination is per 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/accounts"
    querystring = {}
    if type:
        querystring['type'] = type
    if date:
        querystring['date'] = date
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbill(is_id: str, end: str='2018-12-31', start: str='2018-09-17', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single bill."
    id: The ID of the bill.
        end: A date formatted YYYY-MM-DD. If it is added to the request, Firefly III will calculate the appropriate payment and paid dates.

        start: A date formatted YYYY-MM-DD. If it is are added to the request, Firefly III will calculate the appropriate payment and paid dates.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/bills/{is_id}"
    querystring = {}
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exportrecurring(type: str='csv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to export your recurring transactions from Firefly III into a file. Currently supports CSV exports only.
		"
    type: The file type the export file (CSV is currently the only option).
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/data/export/recurring"
    querystring = {}
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchartaccountoverview(end: str, start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the data required to generate a chart with basic asset account balance information.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/chart/account/overview"
    querystring = {'end': end, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightincometag(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', tags: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the income received by the user, grouped by (any) tag.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only deposits to those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        tags: The tags to be included in the results.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/income/tag"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    if tags:
        querystring['tags[]'] = tags
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsinglewebhookmessageattempt(attemptid: int, messageid: int, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When a webhook message fails to send it will store the failure in an "attempt". You can view and analyse these. Webhooks messages that receive too many attempts (failures) will not be fired. You must first clear out old attempts and try again. This endpoint shows you the details of a single attempt. The ID of the attempt must match the corresponding webhook and webhook message."
    attemptid: The webhook attempt ID.
        messageid: The webhook message ID.
        id: The webhook ID.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/webhooks/{is_id}/messages/{messageid}/attempts/{attemptid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightincomecategory(start: str, end: str, categories: str='[
  1,
  2,
  3
]', accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the income received by the user, grouped by (any) category.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        categories: The categories to be included in the results.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only deposits to those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/income/category"
    querystring = {'start': start, 'end': end, }
    if categories:
        querystring['categories[]'] = categories
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrulegroupsac(limit: int=10, query: str='str', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    limit: The number of items returned.
        query: The autocomplete search query.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/autocomplete/rule-groups"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightincomenotag(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the income received by the user, including only income with no tag.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only deposits to those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/income/no-tag"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensetotal(start: str, end: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a sum of the total expenses made by the user.
		"
    start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/total"
    querystring = {'start': start, 'end': end, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbudgetlimitbybudget(is_id: str, start: str='2018-09-17', end: str='2018-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all budget limits for this budget and the money spent, and money left. You can limit the list by submitting a date range as well. The "spent" array for each budget limit is NOT influenced by the start and end date of your query, but by the start and end date of the budget limit itself.
		"
    id: The ID of the requested budget.
        start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/budgets/{is_id}/limits"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbybill(is_id: str, type: str='all', start: str='2018-09-17', end: str='2018-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will list all transactions linked to this bill."
    id: The ID of the bill.
        type: Optional filter on the transaction type(s) returned
        start: A date formatted YYYY-MM-DD.

        end: A date formatted YYYY-MM-DD.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/bills/{is_id}/transactions"
    querystring = {}
    if type:
        querystring['type'] = type
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listavailablebudget(page: int=1, end: str='2018-12-31', start: str='2018-09-17', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Firefly III allows users to set the amount that is available to be budgeted in so-called "available budgets". For example, the user could have 1200,- available to be divided during the coming month. This amount is used on the /budgets page. This endpoint returns all of these amounts and the periods for which they are set.
		"
    page: Page number. The default pagination is 50.
        end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/available_budgets"
    querystring = {}
    if page:
        querystring['page'] = page
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listbillbyobjectgroup(is_id: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all bills with this object group."
    id: The ID of the account.
        page: Page number. The default pagination is per 50 items.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/object_groups/{is_id}/bills"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insighttransfertotal(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a sum of the total amount transfers made by the user.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only transfers between those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/transfer/total"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insightexpensebill(end: str, start: str, accounts: str='[
  1,
  2,
  3
]', bills: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives a summary of the expenses made by the user, grouped by (any) bill.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        accounts: The accounts to be included in the results. If you include ID's of asset accounts or liabilities, only withdrawals from those
asset accounts / liabilities will be included. Other account ID's will be ignored.

        bills: The bills to be included in the results.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/insight/expense/bill"
    querystring = {'end': end, 'start': start, }
    if accounts:
        querystring['accounts[]'] = accounts
    if bills:
        querystring['bills[]'] = bills
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcategory(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all categories."
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/categories"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbasicsummary(end: str, start: str, currency_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns basic sums of the users data, like the net worth, spent and earned amounts. It is multi-currency, and is used in Firefly III to populate the dashboard.
		"
    end: A date formatted YYYY-MM-DD.

        start: A date formatted YYYY-MM-DD.

        currency_code: A currency code like EUR or USD, to filter the result.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/summary/basic"
    querystring = {'end': end, 'start': start, }
    if currency_code:
        querystring['currency_code'] = currency_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpreference(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a single preference and the value."
    name: The name of the preference.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/preferences/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbyrecurrence(is_id: str, start: str='2018-09-17', page: int=1, end: str='2018-09-17', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all transactions created by a recurring transaction, optionally limited to the date ranges specified."
    id: The ID of the recurring transaction.
        start: A date formatted YYYY-MM-DD. Both the start and end date must be present.

        page: Page number. The default pagination is 50.
        end: A date formatted YYYY-MM-DD. Both the start and end date must be present.

        type: Optional filter on the transaction type(s) returned
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/recurrences/{is_id}/transactions"
    querystring = {}
    if start:
        querystring['start'] = start
    if page:
        querystring['page'] = page
    if end:
        querystring['end'] = end
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listrecurrence(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all recurring transactions."
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/recurrences"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listuser(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the users in this instance of Firefly III."
    page: The page number, if necessary. The default pagination is 50, so 50 users per page.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/users"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsingleconfiguration(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns one configuration variable for this Firefly III installation"
    name: The name of the configuration value you want to know.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/configuration/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionbycategory(is_id: str, page: int=1, start: str='2018-09-17', end: str='2018-12-31', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all transactions in a category, optionally limited to the date ranges specified."
    id: The ID of the category.
        page: Page number. The default pagination is per 50.
        start: A date formatted YYYY-MM-DD, to limit the result list.

        end: A date formatted YYYY-MM-DD, to limit the result list.

        type: Optional filter on the transaction type(s) returned
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/categories/{is_id}/transactions"
    querystring = {}
    if page:
        querystring['page'] = page
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrule(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single rule."
    id: The ID of the object.X
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/rules/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuser(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all info of a single user."
    id: The user ID.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/users/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listrule(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all rules."
    page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/rules"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def testrule(is_id: str, start: str='2018-09-17', end: str='2018-09-17', accounts: str='[
  1,
  2,
  3
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test which transactions would be hit by the rule. No changes will be made. Limit the result if you want to."
    id: The ID of the rule.
        start: A date formatted YYYY-MM-DD, to limit the transactions the test will be applied to. Both the start date and the end date must be present.

        end: A date formatted YYYY-MM-DD, to limit the transactions the test will be applied to. Both the start date and the end date must be present.

        accounts: Limit the testing of the rule to these asset accounts or liabilities. Only asset accounts and liabilities will be accepted. Other types will be silently dropped.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/rules/{is_id}/test"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    if accounts:
        querystring['accounts[]'] = accounts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtransactionwithoutcategory(start: str='2018-09-17', type: str='all', end: str='2018-12-31', limit: int=5, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all transactions without a category, possibly limited by start and end"
    start: A date formatted YYYY-MM-DD.

        type: Optional filter on the transaction type(s) returned
        end: A date formatted YYYY-MM-DD.

        limit: Limits the number of results on one page.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/categories/transactions-without-category"
    querystring = {}
    if start:
        querystring['start'] = start
    if type:
        querystring['type'] = type
    if end:
        querystring['end'] = end
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listattachmentbycategory(is_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all attachments."
    id: The ID of the category.
        page: Page number. The default pagination is 50.
        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/categories/{is_id}/attachments"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategory(is_id: str, start: str='2018-09-17', end: str='2018-12-31', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single category."
    id: The ID of the category.
        start: A date formatted YYYY-MM-DD, to show spent and earned info.

        end: A date formatted YYYY-MM-DD, to show spent and earned info.

        
    """
    url = f"https://ff31.p.rapidapi.com/api/v1/categories/{is_id}"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ff31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

