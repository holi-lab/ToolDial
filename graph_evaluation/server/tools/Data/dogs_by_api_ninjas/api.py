import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_dogs(offset: int=None, energy: int=None, protectiveness: int=None, max_weight: int=None, min_weight: int=None, trainability: int=None, shedding: int=None, name: str='golden retriever', min_height: int=None, barking: int=None, max_height: int=None, max_life_expectancy: int=None, min_life_expectancy: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of dog breeds matching specified parameters. Returns at most 20 results. To access more than 20 results, use the offset parameter to offset results in multiple API calls."
    offset: number of results to offset for pagination.
        energy: How much energy the breed has. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates low energy and 5 indicates high energy.
        protectiveness: How likely the breed is to alert strangers. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates minimal alerting and 5 indicates maximum alerting.
        max_weight: maximum weight in pounds.
        min_weight: minimum weight in pounds.
        trainability: How easy it is to train the breed. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates the breed is very difficult to train and 5 indicates the breed is very easy to train.
        shedding: How much hair the breed sheds. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates no shedding and 5 indicates maximum shedding.
        name: name of the breed.
        min_height: minimum height in inches.
        barking: How vocal the breed is. Possible values: 0, 1, 2, 3, 4, 5, where 0 indicates minimal barking and 5 indicates maximum barking.
        max_height: maximum height in inches.
        max_life_expectancy: maximum life expectancy in years.
        min_life_expectancy: minimum life expectancy in years.
        
    """
    url = f"https://dogs-by-api-ninjas.p.rapidapi.com/v1/dogs"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if energy:
        querystring['energy'] = energy
    if protectiveness:
        querystring['protectiveness'] = protectiveness
    if max_weight:
        querystring['max_weight'] = max_weight
    if min_weight:
        querystring['min_weight'] = min_weight
    if trainability:
        querystring['trainability'] = trainability
    if shedding:
        querystring['shedding'] = shedding
    if name:
        querystring['name'] = name
    if min_height:
        querystring['min_height'] = min_height
    if barking:
        querystring['barking'] = barking
    if max_height:
        querystring['max_height'] = max_height
    if max_life_expectancy:
        querystring['max_life_expectancy'] = max_life_expectancy
    if min_life_expectancy:
        querystring['min_life_expectancy'] = min_life_expectancy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dogs-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

