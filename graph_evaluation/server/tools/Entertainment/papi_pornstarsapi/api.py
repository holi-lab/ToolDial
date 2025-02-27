import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listpornstars(max_rank: str=None, min_rank: str=None, max_waist: str=None, max_cup_size: str=None, min_cup_size: str=None, min_weight: str=None, max_weight: str=None, max_age: str=None, ethnicity: str=None, min_age: str=None, nationality: str=None, tattoos: str=None, eyes: str=None, hair: str=None, page: int=None, min_waist: str=None, name: str=None, date_of_birth: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint that allows pornstars to be viewed."
    max_waist: max_waist
        max_cup_size: max_cup_size
        min_cup_size: min_cup_size
        min_weight: min_weight
        max_weight: max_weight
        max_age: max_age
        ethnicity: ethnicity
        min_age: min_age
        nationality: nationality
        tattoos: tattoos
        eyes: eyes
        hair: hair
        page: A page number within the paginated result set.
        min_waist: min_waist
        name: name
        date_of_birth: date_of_birth
        
    """
    url = f"https://papi-pornstarsapi.p.rapidapi.com/pornstars/"
    querystring = {}
    if max_rank:
        querystring['max_rank'] = max_rank
    if min_rank:
        querystring['min_rank'] = min_rank
    if max_waist:
        querystring['max_waist'] = max_waist
    if max_cup_size:
        querystring['max_cup_size'] = max_cup_size
    if min_cup_size:
        querystring['min_cup_size'] = min_cup_size
    if min_weight:
        querystring['min_weight'] = min_weight
    if max_weight:
        querystring['max_weight'] = max_weight
    if max_age:
        querystring['max_age'] = max_age
    if ethnicity:
        querystring['ethnicity'] = ethnicity
    if min_age:
        querystring['min_age'] = min_age
    if nationality:
        querystring['nationality'] = nationality
    if tattoos:
        querystring['tattoos'] = tattoos
    if eyes:
        querystring['eyes'] = eyes
    if hair:
        querystring['hair'] = hair
    if page:
        querystring['page'] = page
    if min_waist:
        querystring['min_waist'] = min_waist
    if name:
        querystring['name'] = name
    if date_of_birth:
        querystring['date_of_birth'] = date_of_birth
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "papi-pornstarsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievepornstar(is_id: str, ethnicity: str=None, min_weight: str=None, max_cup_size: str=None, nationality: str=None, min_waist: str=None, hair: str=None, max_age: str=None, min_age: str=None, min_cup_size: str=None, date_of_birth: str=None, eyes: str=None, max_waist: str=None, name: str=None, tattoos: str=None, max_weight: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint that allows pornstars to be viewed."
    id: A unique integer value identifying this pornstar.
        ethnicity: ethnicity
        min_weight: min_weight
        max_cup_size: max_cup_size
        nationality: nationality
        min_waist: min_waist
        hair: hair
        max_age: max_age
        min_age: min_age
        min_cup_size: min_cup_size
        date_of_birth: date_of_birth
        eyes: eyes
        max_waist: max_waist
        name: name
        tattoos: tattoos
        max_weight: max_weight
        
    """
    url = f"https://papi-pornstarsapi.p.rapidapi.com/pornstars/{is_id}/"
    querystring = {}
    if ethnicity:
        querystring['ethnicity'] = ethnicity
    if min_weight:
        querystring['min_weight'] = min_weight
    if max_cup_size:
        querystring['max_cup_size'] = max_cup_size
    if nationality:
        querystring['nationality'] = nationality
    if min_waist:
        querystring['min_waist'] = min_waist
    if hair:
        querystring['hair'] = hair
    if max_age:
        querystring['max_age'] = max_age
    if min_age:
        querystring['min_age'] = min_age
    if min_cup_size:
        querystring['min_cup_size'] = min_cup_size
    if date_of_birth:
        querystring['date_of_birth'] = date_of_birth
    if eyes:
        querystring['eyes'] = eyes
    if max_waist:
        querystring['max_waist'] = max_waist
    if name:
        querystring['name'] = name
    if tattoos:
        querystring['tattoos'] = tattoos
    if max_weight:
        querystring['max_weight'] = max_weight
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "papi-pornstarsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

