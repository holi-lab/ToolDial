import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_contacts(p: str, json: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all contacts for a given API key."
    p: API Key from [Sms77.io](https://sms77.io).
        json: Decides whether to return response as JSON or CSV (default).
        
    """
    url = f"https://sms77io.p.rapidapi.com/contacts"
    querystring = {'p': p, }
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_status(msg_id: int, p: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a delivery report for a message ID sent with your API key."
    msg_id: The message ID of the SMS.
Can be obtained by setting parameter *JSON*, *return_msg_id* or *details* to **1** when sending SMS via the API.
Alternatively it can be retrieved from the message journal in the user area.
        p: API Key
        
    """
    url = f"https://sms77io.p.rapidapi.com/status"
    querystring = {'msg_id': msg_id, 'p': p, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pricing(p: str, country: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves pricing information for a single country or all."
    p: API Key from [Sms77.io](https://sms77.io).
        country: An ISO Code of the country you wish to retrieve the pricing for.
*Examples*:

-  \"de\" for Germany 
- \"uk\" for Great Britain
- \"fr\" for France

If this parameter is *not* specified, the prices of *all* countries are getting returned.
        format: Whether to return the response as *JSON* or *CSV*. The **default** is *JSON*.
        
    """
    url = f"https://sms77io.p.rapidapi.com/pricing"
    querystring = {'p': p, }
    if country:
        querystring['country'] = country
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_balance(p: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the account balance for the given API key."
    p: Your API key from [Sms77.io](https://sms77.io).
        
    """
    url = f"https://sms77io.p.rapidapi.com/balance"
    querystring = {'p': p, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def outbound(p: str, date_from: str=None, state: str=None, date_to: str=None, is_id: int=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves outbound messages history."
    p: API key from [Sms77.io](https://www.sms77.io)
        date_from: Start date for performed search
        state: Message status - e.g. *completed* / *failed* for **Voice** or *DELIVERED* / *NOTDELIVERED* for **SMS**
        date_to: End date for performed search
        is_id: Message ID
        to: Receiver phone number in any format
        
    """
    url = f"https://sms77io.p.rapidapi.com/journal?type=outbound"
    querystring = {'p': p, }
    if date_from:
        querystring['date_from'] = date_from
    if state:
        querystring['state'] = state
    if date_to:
        querystring['date_to'] = date_to
    if is_id:
        querystring['id'] = is_id
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inbound(p: str, to: str=None, state: str=None, date_from: str=None, date_to: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves inbound messages history."
    p: API key from [Sms77.io](https://www.sms77.io)
        to: Receiver phone number in any format
        state: Message status - e.g. *completed* / *failed* for **Voice** or *DELIVERED* / *NOTDELIVERED* for **SMS**
        date_from: Start date for performed search
        date_to: End date for performed search
        is_id: Message ID
        
    """
    url = f"https://sms77io.p.rapidapi.com/journal?type=inbound"
    querystring = {'p': p, }
    if to:
        querystring['to'] = to
    if state:
        querystring['state'] = state
    if date_from:
        querystring['date_from'] = date_from
    if date_to:
        querystring['date_to'] = date_to
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replies(p: str, is_id: int=None, date_from: str=None, to: str=None, date_to: str=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves message replies history."
    p: API key from [Sms77.io](https://www.sms77.io)
        is_id: Message ID
        date_from: Start date for performed search
        to: Receiver phone number in any format
        date_to: End date for performed search
        state: Message status - e.g. *completed* / *failed* for **Voice** or *DELIVERED* / *NOTDELIVERED* for **SMS**
        
    """
    url = f"https://sms77io.p.rapidapi.com/journal?type=replies"
    querystring = {'p': p, }
    if is_id:
        querystring['id'] = is_id
    if date_from:
        querystring['date_from'] = date_from
    if to:
        querystring['to'] = to
    if date_to:
        querystring['date_to'] = date_to
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read(x_api_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves information regarding subacounts."
    x_api_key: Your sms77 API key for authentication.
        
    """
    url = f"https://sms77io.p.rapidapi.com/subaccounts?action=read"
    querystring = {}
    if x_api_key:
        querystring['X-Api-Key'] = x_api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_analytics(p: str, start: str=None, label: str='all', end: str=None, subaccounts: str='only_main', group_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed statistics of your account directly through our API."
    p: API Key
        start: Start date of the statistics in the format YYYY-MM-DD.
By default, the date of 30 days ago is set.
        label: Shows only data of a specific label.
Allowed values: 'all' (default) or <label>.
        end: End date of the statistics.
By default, the current day.
        subaccounts: Receive the data only for the main account, all your (sub-)accounts or only for specific subaccounts. Allowed values: 'only_main', 'all' or subaccount ID.
        group_by: Defines the grouping of the data.
        
    """
    url = f"https://sms77io.p.rapidapi.com/analytics"
    querystring = {'p': p, }
    if start:
        querystring['start'] = start
    if label:
        querystring['label'] = label
    if end:
        querystring['end'] = end
    if subaccounts:
        querystring['subaccounts'] = subaccounts
    if group_by:
        querystring['group_by'] = group_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_webhooks(p: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all existing webhooks."
    p: API key from [Sms77.io](https://www.sms77.io).
        
    """
    url = f"https://sms77io.p.rapidapi.com/hooks"
    querystring = {'p': p, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def voice(p: str, date_to: str=None, is_id: int=None, state: str=None, to: str=None, date_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves voice messages history."
    p: API key from [Sms77.io](https://www.sms77.io)
        date_to: End date for performed search
        is_id: Message ID
        state: Message status - e.g. *completed* / *failed* etc.
        to: Receiver phone number in any format
        date_from: Start date for performed search
        
    """
    url = f"https://sms77io.p.rapidapi.com/journal?type=voice"
    querystring = {'p': p, }
    if date_to:
        querystring['date_to'] = date_to
    if is_id:
        querystring['id'] = is_id
    if state:
        querystring['state'] = state
    if to:
        querystring['to'] = to
    if date_from:
        querystring['date_from'] = date_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms77io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

