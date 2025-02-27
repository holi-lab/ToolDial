import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_synchronization(is_id: str, accept: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreives the synchronization object to know its status. If succeeded, then the bank account data has been synchronized with the bank and it can be fetched through the /accounts endpoint."
    
    """
    url = f"https://ponto.p.rapidapi.com/synchronizations/{is_id}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ponto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_transactions(accept: str, is_id: str, after: str='ab1be1c1-d00d-47c6-8785-a555a4123101', before: str='ab1be1c1-d00d-47c6-8785-a555a4123101', limit: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List transactions linked to a specific account"
    after: Cursor for pagination. Indicates that the API should return the transaction resources which are immediately after this one in the list (the next page)
        before: Cursor for pagination. Indicates that the API should return the transaction resources which are immediately before this one in the list (the previous page)
        
    """
    url = f"https://ponto.p.rapidapi.com/accounts/{is_id}/transactions"
    querystring = {'accept': accept, }
    if after:
        querystring['after'] = after
    if before:
        querystring['before'] = before
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ponto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_accounts(accept: str, after: str='953934eb-229a-4fd2-8675-07794078cc7d', before: str='953934eb-229a-4fd2-8675-07794078cc7d', limit: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the accounts available in this integration"
    after: Cursor for pagination. Indicates that the API should return the account resources which are immediately after this one in the list (the next page)
        before: Cursor for pagination. Indicates that the API should return the account resources which are immediately before this one in the list (the previous page)
        
    """
    url = f"https://ponto.p.rapidapi.com/accounts"
    querystring = {'accept': accept, }
    if after:
        querystring['after'] = after
    if before:
        querystring['before'] = before
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ponto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

