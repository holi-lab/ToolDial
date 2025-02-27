import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_messages(created_at: str='2022-08-19', order_by: str='created_at', to: str='+3579865333', order_direction: str='desc', status: str='REJECTED', limit: int=15, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch paginated messages.
		
		Please note that the messages will be sorted by the date of creating (sending) - latest first."
    created_at: Filter by created at
        order_by: Field to sort by
        to: Filter by recipient number
        order_direction: Sort messages.
        status: Filter by status
        limit: The number of messages per page.
        
    """
    url = f"https://smsto.p.rapidapi.com/messages"
    querystring = {}
    if created_at:
        querystring['created_at'] = created_at
    if order_by:
        querystring['order_by'] = order_by
    if to:
        querystring['to'] = to
    if order_direction:
        querystring['order_direction'] = order_direction
    if status:
        querystring['status'] = status
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_campaign(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last campaign that you have sent"
    
    """
    url = f"https://smsto.p.rapidapi.com/last/campaign"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_message_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the message that you have sent by ID"
    
    """
    url = f"https://smsto.p.rapidapi.com/message/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smsto.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

