import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def read_checkout(checkout_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the checkout by ID."
    
    """
    url = f"https://picatic.p.rapidapi.com/checkout/{checkout_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "picatic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def my_tickets(event_id: str, limit: str, offset: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all ticket prices for a particular event."
    
    """
    url = f"https://picatic.p.rapidapi.com/ticket_price?filter[event_id]={event_id}&page[limit]={limit}&page[offset]={offset}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "picatic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apply_promo_code(event_id: str, promo_code: str, limit: str, offset: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Apply a promo code to a ticket price - (Optional)"
    
    """
    url = f"https://picatic.p.rapidapi.com/ticket_price?filter[event_id]={event_id}&tpd_code={promo_code}&include=ticket_price_discount&page[limit]={limit}&page[offset]={offset}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "picatic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def my_events(limit: int, offset: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of your events."
    
    """
    url = f"https://picatic.p.rapidapi.com/event?filter[user_id]=me&filter[status]=active&page[limit]={limit}&page[offset]={offset}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "picatic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

