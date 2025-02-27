import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mohd_hamza_copy(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Indian Railways API Endpoints Let us now take a detailed look at the Indian Railways API endpoints. PNR Status Check PNR stands for Passenger Name Record and is a unique number issued by the Indian Railways system per booking. The PNR status check endpoint returns the booking status of a particular PNR number."
    
    """
    url = f"https://railways-travel.p.rapidapi.com/www.sparrowtravel.in"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "railways-travel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mohd_hamza(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Indian Railways API Endpoints Let us now take a detailed look at the Indian Railways API endpoints. PNR Status Check PNR stands for Passenger Name Record and is a unique number issued by the Indian Railways system per booking. The PNR status check endpoint returns the booking status of a particular PNR number."
    
    """
    url = f"https://railways-travel.p.rapidapi.com/www.sparrowtravel.in"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "railways-travel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

