import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_call_detail(account_id: str, callid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://ctm.p.rapidapi.com/accounts/{account_id}/calls/{callid}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delete_delete_a_webhook(account_id: str, webhook_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://ctm.p.rapidapi.com/accounts/{account_id}/webhooks/{webhook_id}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_calls(account_id: str, page: str=None, filter: str=None, start_date: str=None, end_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Calls API allows you easy access to all your call data. You can subscribe to new incoming calls to receive an HTTP request when a call is completed. You can also query the call API to retrieve all past calls."
    page: the page offset to query
        filter: a search string to look for calls with specific callerid, caller_number, called_number, source name, etc...
        start_date: a starting date offset to query
        end_date: a ending date offset to query
        
    """
    url = f"https://ctm.p.rapidapi.com/accounts/{account_id}/calls.json"
    querystring = {}
    if page:
        querystring['page'] = page
    if filter:
        querystring['filter'] = filter
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_numbers(aid: str, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The numbers API allows you to search, purchase, and manage settings on each tracking number."
    page: the page offset to query
        
    """
    url = f"https://ctm.p.rapidapi.com/v1/accounts/{aid}/numbers.json"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_text_messages(aid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The SMS API allows you to search, send, and manage text messaging triggers for all capable numbers."
    
    """
    url = f"https://ctm.p.rapidapi.com/accounts/{aid}/sms.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_accounts(page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The accounts API allows you to create and manage accounts."
    
    """
    url = f"https://ctm.p.rapidapi.com/accounts.json"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_users(aid: str, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The users API allows you to create and manage users."
    page: the page offset to query
        
    """
    url = f"https://ctm.p.rapidapi.com/accounts/{aid}/users.json"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_numbers(area_code: str, country_code: str, account_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    area_code: an area code to search and match on
        country_code: the country where you would like to purchase this number. currently supported countries include "US" and "CA"
        
    """
    url = f"https://ctm.p.rapidapi.com/accounts/{account_id}/numbers/search.json"
    querystring = {'area_code': area_code, 'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ctm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

