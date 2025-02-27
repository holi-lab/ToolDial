import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_trips(date: str, from_id: str, passenger0_age: int, time: str, to_id: str, passenger3_age: int=None, passenger2_discount: str=None, passenger4_discount: str=None, passenger1_discount: str=None, passenger3_discount: str=None, passenger2_age: int=None, passenger0_discount: str=None, passenger4_age: int=None, passenger1_age: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find trips between two stations"
    date: Travel date in format DD.MM.YYYY
        from_id: Departure id
        passenger0_age: Age of the first passenger
        time: Travel time
        to_id: Arrival id
        passenger3_age: Age of the fourth passenger
        passenger2_discount: Discount of the third passenger
        passenger4_discount: Discount of the fifth passenger
        passenger1_discount: Discount of the second passenger
        passenger3_discount: Discount of the fourth passenger
        passenger2_age: Age of the third passenger
        passenger0_discount: Discount of the first passenger
        passenger4_age: Age of the fifth passenger
        passenger1_age: Age of the second passenger
        
    """
    url = f"https://deutsche-bahn1.p.rapidapi.com/trips"
    querystring = {'date': date, 'from_id': from_id, 'passenger0_age': passenger0_age, 'time': time, 'to_id': to_id, }
    if passenger3_age:
        querystring['passenger3_age'] = passenger3_age
    if passenger2_discount:
        querystring['passenger2_discount'] = passenger2_discount
    if passenger4_discount:
        querystring['passenger4_discount'] = passenger4_discount
    if passenger1_discount:
        querystring['passenger1_discount'] = passenger1_discount
    if passenger3_discount:
        querystring['passenger3_discount'] = passenger3_discount
    if passenger2_age:
        querystring['passenger2_age'] = passenger2_age
    if passenger0_discount:
        querystring['passenger0_discount'] = passenger0_discount
    if passenger4_age:
        querystring['passenger4_age'] = passenger4_age
    if passenger1_age:
        querystring['passenger1_age'] = passenger1_age
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deutsche-bahn1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for train and public transport stations in Germany and across Europe"
    query: Query parameter
        
    """
    url = f"https://deutsche-bahn1.p.rapidapi.com/autocomplete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deutsche-bahn1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

