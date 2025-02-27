import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetchspecificcontact(is_id: int, custom_fields: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific contact"
    id: ID of contact to fetch
        custom_fields: If set, custom fields will be displayed in the output
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/contacts/{is_id}"
    querystring = {}
    if custom_fields:
        querystring['custom_fields'] = custom_fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcustomfields(is_id: int, start_id: int=None, limit: int=None, end_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get custom fields"
    id: ID of contact
        start_id: Object ID to fetch next page
        limit: Amount of records to return. 25 by default.
        end_id: Object ID to fetch previous page
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/contacts/{is_id}/custom_fields"
    querystring = {}
    if start_id:
        querystring['start_id'] = start_id
    if limit:
        querystring['limit'] = limit
    if end_id:
        querystring['end_id'] = end_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchwebhooks(url: int=None, start_id: int=None, end_id: int=None, limit: int=None, sort: int=None, kind: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all webhooks"
    url: Filter. Filter collection by url
        start_id: Object ID to fetch next page
        end_id: Object ID to fetch previous page
        limit: Amount of records to return. 25 by default.
        sort: Sort field. Available fields: url, kind
        kind: Filter. Filter collection by kind
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/webhooks"
    querystring = {}
    if url:
        querystring['url'] = url
    if start_id:
        querystring['start_id'] = start_id
    if end_id:
        querystring['end_id'] = end_id
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if kind:
        querystring['kind'] = kind
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchoauthauthorizedapplications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Oauth authorized applications"
    
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/oauth/authorized_applications"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcampaigns(min_start_date: int=None, max_updated_datetime: int=None, sort: int=None, min_created_datetime: int=None, limit: int=None, max_created_datetime: int=None, end_id: int=None, name: int=None, max_start_date: int=None, min_updated_datetime: int=None, language: int=None, status: int=None, start_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all campaigns"
    min_start_date: Filter. Format: YYYY-MM-DD. Filter collection by min_start_date, required max_start_date too.
        max_updated_datetime: Filter. Format: YYYY-MM-DDTHH:MM:SS. Filter collection by max_updated_datetime, required min_updated_datetime too.
        sort: Sort field. Available fields: name, status, kind, created_at, start_date
        min_created_datetime: Filter. Format: YYYY-MM-DDTHH:MM:SS. Filter collection by min_created_datetime, required max_created_datetime too.
        limit: Amount of records to return. 25 by default.
        max_created_datetime: Filter. Format: YYYY-MM-DDTHH:MM:SS. Filter collection by max_created_datetime, required min_created_datetime too.
        end_id: Object ID to fetch previous page
        name: Filter. Filter collection by name
        max_start_date: Filter. Format: YYYY-MM-DD. Filter collection by max_start_date, required mix_start_date too.
        min_updated_datetime: Filter. Format: YYYY-MM-DDTHH:MM:SS. Filter collection by min_updated_datetime, required max_updated_datetime too.
        language: Filter. Filter collection by language
        status: Filter. Filter collection by status
        start_id: Object ID to fetch next page
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/campaigns"
    querystring = {}
    if min_start_date:
        querystring['min_start_date'] = min_start_date
    if max_updated_datetime:
        querystring['max_updated_datetime'] = max_updated_datetime
    if sort:
        querystring['sort'] = sort
    if min_created_datetime:
        querystring['min_created_datetime'] = min_created_datetime
    if limit:
        querystring['limit'] = limit
    if max_created_datetime:
        querystring['max_created_datetime'] = max_created_datetime
    if end_id:
        querystring['end_id'] = end_id
    if name:
        querystring['name'] = name
    if max_start_date:
        querystring['max_start_date'] = max_start_date
    if min_updated_datetime:
        querystring['min_updated_datetime'] = min_updated_datetime
    if language:
        querystring['language'] = language
    if status:
        querystring['status'] = status
    if start_id:
        querystring['start_id'] = start_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcallsforcampaign(is_id: int, status: int=None, limit: int=None, start_id: int=None, end_id: int=None, caller_name: int=None, sort: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all calls for a campaign"
    id: ID of campaign
        status: Filter. Filter collection by status
        limit: Amount of records to return. 25 by default.
        start_id: Object ID to fetch next page
        end_id: Object ID to fetch previous page
        caller_name: Filter. Filter collection by caller name
        sort: Sort field. Available fields: status, called_at
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/campaigns/{is_id}/calls"
    querystring = {}
    if status:
        querystring['status'] = status
    if limit:
        querystring['limit'] = limit
    if start_id:
        querystring['start_id'] = start_id
    if end_id:
        querystring['end_id'] = end_id
    if caller_name:
        querystring['caller_name'] = caller_name
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchwebhook(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific webhook"
    id: ID of webhook
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/webhooks/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcontactanswers(is_id: int, end_id: int=None, start_id: int=None, limit: int=None, result: int=None, sort: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get contact's answers"
    id: ID of contact
        end_id: Object ID to fetch previous page
        start_id: Object ID to fetch next page
        limit: Amount of records to return. 25 by default.
        result: Filter. Filter collection by result
        sort: Sort field. Available fields: answer_type, created_at
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/contacts/{is_id}/answers"
    querystring = {}
    if end_id:
        querystring['end_id'] = end_id
    if start_id:
        querystring['start_id'] = start_id
    if limit:
        querystring['limit'] = limit
    if result:
        querystring['result'] = result
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcampaign(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific campaign"
    id: ID of campaign
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/campaigns/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchquestionanswers(is_id: int, start_id: int=None, limit: int=None, end_id: int=None, sort: int=None, result: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get question's answers"
    id: ID of question
        start_id: Object ID to fetch next page
        limit: Amount of records to return. 25 by default.
        end_id: Object ID to fetch previous page
        sort: Sort field. Available fields: created_at
        result: Filter. Filter collection by result
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/questions/{is_id}/answers"
    querystring = {}
    if start_id:
        querystring['start_id'] = start_id
    if limit:
        querystring['limit'] = limit
    if end_id:
        querystring['end_id'] = end_id
    if sort:
        querystring['sort'] = sort
    if result:
        querystring['result'] = result
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcontacts(is_id: int, sort: int=None, last_name: int=None, urgency: int=None, start_id: int=None, name: int=None, company: int=None, field_id: int=None, phone: int=None, email: int=None, custom_fields: int=None, limit: int=None, end_id: int=None, status: int=None, first_name: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get contacts for a campaign"
    id: ID of campaign
        sort: Sort field. Available fields: last_name, first_name, company, calls.status
                             calls.called_at, calls.caller.user.first_name, calls.caller.user.last_name, urgency
        last_name: Filter. Filter collection by last_name
        urgency: Filter. Filter collection by urgency
        start_id: Object ID to fetch next page
        name: Filter. Filter collection by name
        company: Filter. Filter collection by company
        field_id: Filter. Filter collection by field_id
        phone: Filter. Filter collection by phone
        email: Filter. Filter collection by email
        custom_fields: If set, custom fields will be displayed in the output
        limit: Amount of records to return. 25 by default.
        end_id: Object ID to fetch previous page
        status: Filter. Filter collection by status
        first_name: Filter. Filter collection by first_name
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/campaigns/{is_id}/contacts"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if last_name:
        querystring['last_name'] = last_name
    if urgency:
        querystring['urgency'] = urgency
    if start_id:
        querystring['start_id'] = start_id
    if name:
        querystring['name'] = name
    if company:
        querystring['company'] = company
    if field_id:
        querystring['field_id'] = field_id
    if phone:
        querystring['phone'] = phone
    if email:
        querystring['email'] = email
    if custom_fields:
        querystring['custom_fields'] = custom_fields
    if limit:
        querystring['limit'] = limit
    if end_id:
        querystring['end_id'] = end_id
    if status:
        querystring['status'] = status
    if first_name:
        querystring['first_name'] = first_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcampaignquestions(is_id: int, start_id: int=None, end_id: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get campaign's questions"
    id: ID of campaign
        start_id: Object ID to fetch next page
        end_id: Object ID to fetch previous page
        limit: Amount of records to return. 25 by default.
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/campaigns/{is_id}/questions"
    querystring = {}
    if start_id:
        querystring['start_id'] = start_id
    if end_id:
        querystring['end_id'] = end_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchcalls(status: int=None, caller_name: int=None, sort: int=None, start_id: int=None, limit: int=None, end_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all calls"
    status: Filter. Filter collection by status
        caller_name: Filter. Filter collection by caller name
        sort: Sort field. Available fields: status, called_at
        start_id: Object ID to fetch next page
        limit: Amount of records to return. 25 by default.
        end_id: Object ID to fetch previous page
        
    """
    url = f"https://upcall-upcall-v1.p.rapidapi.com/calls"
    querystring = {}
    if status:
        querystring['status'] = status
    if caller_name:
        querystring['caller_name'] = caller_name
    if sort:
        querystring['sort'] = sort
    if start_id:
        querystring['start_id'] = start_id
    if limit:
        querystring['limit'] = limit
    if end_id:
        querystring['end_id'] = end_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "upcall-upcall-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

