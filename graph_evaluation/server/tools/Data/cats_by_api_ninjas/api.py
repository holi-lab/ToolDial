import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_cats(max_life_expectancy: int=None, playfulness: int=None, grooming: int=None, min_life_expectancy: int=None, max_weight: int=None, family_friendly: int=None, min_weight: int=None, offset: int=None, children_friendly: int=None, other_pets_friendly: int=None, shedding: int=None, name: str='aegean', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Cats API endpoint."
    max_life_expectancy: maximum life expectancy in years.
        playfulness: How playful the cat is. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates serious and stern and 5 indicates maximum playfulness.
        grooming: How much work is required to properly groom the cat. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates maximum grooming effort and 5 indicates minimum grooming effort.
        min_life_expectancy: minimum life expectancy in years.
        max_weight: maximum weight in pounds.
        family_friendly: How affectionate the cat is to family. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates minimal affection and 5 indicates maximum affection.
        min_weight: minimum weight in pounds.
        offset: number of results to offset for pagination.
        children_friendly: How well the cat gets along with children. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates the cat does not get along well with kids and 5 indicates the cat is very kid-friendly.
        other_pets_friendly: How well the cat gets along with other pets in the household (for example, dogs). Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates the cat isn't very friendly to other pets and 5 indicates the cat gets along very well with other pets.
        shedding: How much hair the cat sheds. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates no shedding and 5 indicates maximum shedding.
        name: the name of cat breed.
        
    """
    url = f"https://cats-by-api-ninjas.p.rapidapi.com/v1/cats"
    querystring = {}
    if max_life_expectancy:
        querystring['max_life_expectancy'] = max_life_expectancy
    if playfulness:
        querystring['playfulness'] = playfulness
    if grooming:
        querystring['grooming'] = grooming
    if min_life_expectancy:
        querystring['min_life_expectancy'] = min_life_expectancy
    if max_weight:
        querystring['max_weight'] = max_weight
    if family_friendly:
        querystring['family_friendly'] = family_friendly
    if min_weight:
        querystring['min_weight'] = min_weight
    if offset:
        querystring['offset'] = offset
    if children_friendly:
        querystring['children_friendly'] = children_friendly
    if other_pets_friendly:
        querystring['other_pets_friendly'] = other_pets_friendly
    if shedding:
        querystring['shedding'] = shedding
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cats-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

