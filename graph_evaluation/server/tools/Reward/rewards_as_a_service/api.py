import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_4_get_list_of_rewards(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the list of rewards available for the platform"
    
    """
    url = f"https://tangocard-rewards-as-a-service.p.rapidapi.com/rewards"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tangocard-rewards-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_get_account_information(customer: str, account_identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the information for an account created under a Customer"
    
    """
    url = f"https://tangocard-rewards-as-a-service.p.rapidapi.com/accounts/{customer}/{account_identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tangocard-rewards-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_get_order_information(order_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more information about an order"
    
    """
    url = f"https://tangocard-rewards-as-a-service.p.rapidapi.com/orders/{order_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tangocard-rewards-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_7_get_order_history(customer: str, account_identifier: str, offset: int=None, limit: int=None, start_date: str=None, end_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more information about all the orders placed on this customer and account"
    
    """
    url = f"https://tangocard-rewards-as-a-service.p.rapidapi.com/orders"
    querystring = {'customer': customer, 'account_identifier': account_identifier, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tangocard-rewards-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

