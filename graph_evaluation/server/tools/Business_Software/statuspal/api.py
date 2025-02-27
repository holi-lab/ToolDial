import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getstatuspagestatus(statuspagesubdomain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    statuspagesubdomain: The subdomain of your status page (subdomain.statuspal.io)
        
    """
    url = f"https://statuspal.p.rapidapi.com/status_pages/{statuspagesubdomain}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hello(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://statuspal.p.rapidapi.com/hello"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getservicestatus(serviceid: int, statuspagesubdomain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    serviceid: The ID of a service
        statuspagesubdomain: The subdomain of your status page (subdomain.statuspal.io)
        
    """
    url = f"https://statuspal.p.rapidapi.com/status_pages/{statuspagesubdomain}/services/{serviceid}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhooks(statuspagesubdomain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    statuspagesubdomain: The subdomain of your status page (subdomain.statuspal.io)
        
    """
    url = f"https://statuspal.p.rapidapi.com/status_pages/{statuspagesubdomain}/webhooks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebhook(webhookid: int, statuspagesubdomain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    webhookid: The ID of a webhook
        statuspagesubdomain: The subdomain of your status page (subdomain.statuspal.io)
        
    """
    url = f"https://statuspal.p.rapidapi.com/status_pages/{statuspagesubdomain}/webhooks/{webhookid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubscription(statuspagesubdomain: str, subscriptionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    statuspagesubdomain: The subdomain of your status page (subdomain.statuspal.io)
        subscriptionid: The ID of a subscription
        
    """
    url = f"https://statuspal.p.rapidapi.com/status_pages/{statuspagesubdomain}/subscriptions/{subscriptionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubscriptions(statuspagesubdomain: str, limit: int=None, after: str=None, before: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    statuspagesubdomain: The subdomain of your status page (subdomain.statuspal.io)
        limit: Set the number of subscriptions to return in the response.
This defaults to 20 items, and can be a maximum of 100.

        after: Used as a cursor for pagination.
        before: Used as a cursor for pagination.
        
    """
    url = f"https://statuspal.p.rapidapi.com/status_pages/{statuspagesubdomain}/subscriptions"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if after:
        querystring['after'] = after
    if before:
        querystring['before'] = before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getincidents(statuspagesubdomain: str, limit: int=None, before: str=None, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    statuspagesubdomain: The subdomain of your status page (subdomain.statuspal.io)
        limit: Set the number of incidents to return in the response.
This defaults to 20 items, and can be a maximum of 100.

        before: Used as a cursor for pagination.
        after: Used as a cursor for pagination.
        
    """
    url = f"https://statuspal.p.rapidapi.com/status_pages/{statuspagesubdomain}/incidents"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if before:
        querystring['before'] = before
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statuspal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

