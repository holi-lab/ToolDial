import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def inaudi_php_number_number_decimal_decimal_display_display(decimal: int, number: str, display: int=1000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get any decimal"
    decimal: The n-th decimal that you want to know - 0 < n <= 10.000.000.000 if number is Pi else 0 < n <= 1.000.000.000
        number: The irrational number (pi - e - sqrt2 - ln2 - ln10 - apery - phi - mascheroni). The default number is pi
        display: Display the n next decimals - 0 < n <= 1000 - This parameter is optional
        
    """
    url = f"https://totodunet-decimals-search-v1.p.rapidapi.com/inaudi.php?number={number}&decimal={decimal}&display={display}"
    querystring = {}
    if display:
        querystring['display'] = display
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "totodunet-decimals-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inaudi_php_number_number_search_search_start_start_end_end(number: str, search: int, end: int=1000000000, start: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the first position of any numeric string"
    number: The irrational number (pi - e - sqrt2 - ln2 - ln10 - apery - phi - mascheroni). The default number is pi
        search: The numeric string that you want to know his first position in decimals
        end: The limit end (the default value is max value : 1.000.000.000)
        start: Start to search (0 < n < end value) - the default value is 1
        
    """
    url = f"https://totodunet-decimals-search-v1.p.rapidapi.com/inaudi.php?number={number}&search={search}&start={start}&end={end}"
    querystring = {}
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "totodunet-decimals-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inaudi_php_number_number_count_count_start_start_end_end(count: int, number: str, start: int=1, end: int=1000000000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Count your number in decimals between start and end"
    count: The numeric string that you want count his occurency
        number: The irrational number (pi - e - sqrt2 - ln2 - ln10 - zeta3 - phi - mascheroni). The default number is pi
        start: Start to count (0 < n < end value) - the default value is 1
        end: The limit end (the default value is max value : 1.000.000.000)
        
    """
    url = f"https://totodunet-decimals-search-v1.p.rapidapi.com/inaudi.php?number={number}&count={count}&start={start}&end={end}"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "totodunet-decimals-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

