import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_identities(userid: str=None, sourceid: str=None, cursor: str=None, size: int=10, accountid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's identity profiles from employment data sources."
    userid: Filter to those associated with a particular user ID.
        sourceid: Filter to those associated with a particular source ID.
        cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        accountid: Filter to those associated with a particular account ID.
        
    """
    url = f"https://smile2.p.rapidapi.com/identities"
    querystring = {}
    if userid:
        querystring['userId'] = userid
    if sourceid:
        querystring['sourceId'] = sourceid
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    if accountid:
        querystring['accountId'] = accountid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_invites(cursor: str=None, size: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of invites sent out to users."
    cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/invites"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_incomes(enddate: str=None, userid: str=None, sourceid: str=None, size: int=10, startdate: str=None, cursor: str=None, accountid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's incomes from employment data sources."
    enddate: Filter by income date, end of date range (YYYY-MM-DD)
        userid: Filter to those associated with a particular user ID.
        sourceid: Filter to those associated with a particular source ID.
        size: The number of objects you want returned in a collection.
        startdate: Filter by income date, start of date range (YYYY-MM-DD)
        cursor: Uses the filter values of the previous page to determine the next set of items.
        accountid: Filter to those associated with a particular account ID.
        
    """
    url = f"https://smile2.p.rapidapi.com/incomes"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if userid:
        querystring['userId'] = userid
    if sourceid:
        querystring['sourceId'] = sourceid
    if size:
        querystring['size'] = size
    if startdate:
        querystring['startDate'] = startdate
    if cursor:
        querystring['cursor'] = cursor
    if accountid:
        querystring['accountId'] = accountid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a particular user profile."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/users/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_invite(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get invite sent to users by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/invites/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_identity(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's identity profile by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/identities/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_providers(enabled: bool=None, type: str=None, cursor: str=None, size: int=10, active: bool=None, accountconnection: bool=None, subtype: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of employment data providers from the Smile Network."
    enabled: Filter results to enabled or disabled providers.
        type: Filter results to certain type of providers.
        cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        active: Filter results to active or inactive providers.
        accountconnection: Filter results to providers with real-time account connections.
        subtype: Filter results to certain subset of a certain subtype of providers
        name: Filter results by provider name
        
    """
    url = f"https://smile2.p.rapidapi.com/providers"
    querystring = {}
    if enabled:
        querystring['enabled'] = enabled
    if type:
        querystring['type'] = type
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    if active:
        querystring['active'] = active
    if accountconnection:
        querystring['accountConnection'] = accountconnection
    if subtype:
        querystring['subType'] = subtype
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_employments(sourceid: str=None, accountid: str=None, cursor: str=None, size: int=10, enddate: str=None, userid: str=None, startdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's previous employments."
    sourceid: Filter to those associated with a particular source ID.
        accountid: Filter to those associated with a particular account ID.
        cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        enddate: Filter by employment date, end of date range (YYYY-MM-DD)
        userid: Filter to those associated with a particular user ID.
        startdate: Filter by employment date, start of date range (YYYY-MM-DD)
        
    """
    url = f"https://smile2.p.rapidapi.com/employments"
    querystring = {}
    if sourceid:
        querystring['sourceId'] = sourceid
    if accountid:
        querystring['accountId'] = accountid
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    if enddate:
        querystring['endDate'] = enddate
    if userid:
        querystring['userId'] = userid
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_webhooks(cursor: str=None, size: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all webhook endpoints."
    cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/webhooks"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_ratings(cursor: str=None, size: int=10, sourceid: str=None, userid: str=None, accountid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's ratings from employment data sources."
    cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        sourceid: Filter to those associated with a particular source ID.
        userid: Filter to those associated with a particular user ID.
        accountid: Filter to those associated with a particular account ID.
        
    """
    url = f"https://smile2.p.rapidapi.com/ratings"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    if sourceid:
        querystring['sourceId'] = sourceid
    if userid:
        querystring['userId'] = userid
    if accountid:
        querystring['accountId'] = accountid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rating(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's ratings by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/ratings/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_provider(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a data provider from the Smile Network."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/providers/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_users(enddate: str='2021-04-21', size: int=10, startdate: str='2021-04-01', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List users from the Smile Network."
    enddate: Filter by user creation date, end of date range (YYYY-MM-DD)
        size: The number of objects you want returned in a collection.
        startdate: Filter by user creation date, start of date range (YYYY-MM-DD)
        cursor: Uses the filter values of the previous page to determine the next set of items.
        
    """
    url = f"https://smile2.p.rapidapi.com/users"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if size:
        querystring['size'] = size
    if startdate:
        querystring['startDate'] = startdate
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_invite_template(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get invite template."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/inviteTemplates/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_contributions(cursor: str=None, accountid: str=None, sourceid: str=None, startdate: str=None, size: int=10, enddate: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's social security contributions from employment data sources."
    cursor: Uses the filter values of the previous page to determine the next set of items.
        accountid: Filter to those associated with a particular account ID.
        sourceid: Filter to those associated with a particular source ID.
        startdate: Filter by contribution date, start of date range (YYYY-MM-DD)
        size: The number of objects you want returned in a collection.
        enddate: Filter by contribution date, end of date range (YYYY-MM-DD)
        userid: Filter to those associated with a particular user ID.
        
    """
    url = f"https://smile2.p.rapidapi.com/contributions"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if accountid:
        querystring['accountId'] = accountid
    if sourceid:
        querystring['sourceId'] = sourceid
    if startdate:
        querystring['startDate'] = startdate
    if size:
        querystring['size'] = size
    if enddate:
        querystring['endDate'] = enddate
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_upload(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's upload by ID (deprecated)."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/uploads/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_income(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's income by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/incomes/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_contribution(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's contribution by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/contributions/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_webhook(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a webhook endpoint."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/webhooks/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_uploads(size: int=10, userid: str=None, startdate: str='2021-04-01', enddate: str='2021-04-21', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's uploads (deprecated)."
    size: The number of objects you want returned in a collection.
        userid: Filter to those associated with a particular user ID.
        startdate: Filter by upload date, start of date range (YYYY-MM-DD)
        enddate: Filter by upload date, end of date range (YYYY-MM-DD)
        cursor: Uses the filter values of the previous page to determine the next set of items.
        
    """
    url = f"https://smile2.p.rapidapi.com/uploads"
    querystring = {}
    if size:
        querystring['size'] = size
    if userid:
        querystring['userId'] = userid
    if startdate:
        querystring['startDate'] = startdate
    if enddate:
        querystring['endDate'] = enddate
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_invite_templates(size: int=10, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of invite templates."
    size: The number of objects you want returned in a collection.
        cursor: Uses the filter values of the previous page to determine the next set of items.
        
    """
    url = f"https://smile2.p.rapidapi.com/inviteTemplates"
    querystring = {}
    if size:
        querystring['size'] = size
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_employment(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's employment by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/employments/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transaction(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's transaction by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/transactions/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_transactions(cursor: str=None, size: int=10, accountid: str=None, enddate: str='2021-04-21', startdate: str='2021-04-01', userid: str=None, sourceid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's transactions from employment data sources."
    cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        accountid: Filter to those associated with a particular account ID.
        enddate: Filter by transaction date, end of date range (YYYY-MM-DD)
        startdate: Filter by transaction date, start of date range (YYYY-MM-DD)
        userid: Filter to those associated with a particular user ID.
        sourceid: Filter to those associated with a particular source ID.
        
    """
    url = f"https://smile2.p.rapidapi.com/transactions"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    if accountid:
        querystring['accountId'] = accountid
    if enddate:
        querystring['endDate'] = enddate
    if startdate:
        querystring['startDate'] = startdate
    if userid:
        querystring['userId'] = userid
    if sourceid:
        querystring['sourceId'] = sourceid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_archives(cursor: str=None, enddate: str='2021-04-21', userid: str=None, size: int=10, startdate: str='2021-04-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's archives."
    cursor: Uses the filter values of the previous page to determine the next set of items.
        enddate: Filter by archive date, end of date range (YYYY-MM-DD)
        userid: Filter to those associated with a particular user ID.
        size: The number of objects you want returned in a collection.
        startdate: Filter by archive date, start of date range (YYYY-MM-DD)
        
    """
    url = f"https://smile2.p.rapidapi.com/archives"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if enddate:
        querystring['endDate'] = enddate
    if userid:
        querystring['userId'] = userid
    if size:
        querystring['size'] = size
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_account(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's particular account information."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/accounts/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_archive(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's archive by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/archives/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_file_types(cursor: str=None, size: int=10, enabled: bool=None, ocr: bool=None, active: bool=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the available file types for uploads."
    cursor: Uses the filter values of the previous page to determine the next set of items.
        size: The number of objects you want returned in a collection.
        enabled: Filter results to enabled or disabled file types.
        ocr: Filter results to file types that support OCR.
        active: Filter results to active or inactive file types.
        type: Filter results to certain type of file types.
        
    """
    url = f"https://smile2.p.rapidapi.com/fileTypes"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if size:
        querystring['size'] = size
    if enabled:
        querystring['enabled'] = enabled
    if ocr:
        querystring['ocr'] = ocr
    if active:
        querystring['active'] = active
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_accounts(enddate: str='2021-04-21', cursor: str=None, startdate: str='2021-04-01', userid: str=None, size: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of user accounts."
    enddate: Filter by account linking date, end of date range (YYYY-MM-DD)
        cursor: Uses the filter values of the previous page to determine the next set of items.
        startdate: Filter by account linking date, start of date range (YYYY-MM-DD)
        userid: Filter to those associated with a particular user ID.
        size: The number of objects you want returned in a collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/accounts"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if cursor:
        querystring['cursor'] = cursor
    if startdate:
        querystring['startDate'] = startdate
    if userid:
        querystring['userId'] = userid
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_documents(size: int=10, cursor: str=None, accountid: str=None, userid: str=None, sourceid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the user's documents from employment data sources."
    size: The number of objects you want returned in a collection.
        cursor: Uses the filter values of the previous page to determine the next set of items.
        accountid: Filter to those associated with a particular account ID.
        userid: Filter to those associated with a particular user ID.
        sourceid: Filter to those associated with a particular source ID.
        
    """
    url = f"https://smile2.p.rapidapi.com/documents"
    querystring = {}
    if size:
        querystring['size'] = size
    if cursor:
        querystring['cursor'] = cursor
    if accountid:
        querystring['accountId'] = accountid
    if userid:
        querystring['userId'] = userid
    if sourceid:
        querystring['sourceId'] = sourceid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_document(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a user's document by ID."
    id: ID of the specific object in the collection.
        
    """
    url = f"https://smile2.p.rapidapi.com/documents/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smile2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

