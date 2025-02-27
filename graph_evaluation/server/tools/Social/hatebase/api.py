import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hatebase_api(increment_subincrement: str, get_32_digit_key: str, first_key_first_value_second_key_second_value: str, first_key: str, first_value: str, second_key: str, output: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Hatebase API allows authorized users to retrieve Hatebase data asynchronously for use in their own applications. The API is a continual work in progress, so as new vesions of the API become available, old versions will be retired (after a suitable transition period)."
    increment: Version increment
        subincrement: Version sub-increment
        get_32_digit_key: A private API key provisioned at hatebase.org
        first_key: Filters are key/value pairs which correspond to output parameters. This is the first pair's key.
        first_value: This is the first pair's value.
        second_key: This is the second pair's key.
        second_value: This is the second pair's value.
        output: Possible options: json, xml.
        query: Possible options: vocabulary, sightings
        
    """
    url = f"https://hatebase-hatebase.p.rapidapi.com/v{increment}-{subincrement}/{32_digit_key}/{query}/{output}/{first_key}={first_value}|{second_key}={second_value}"
    querystring = {'increment-subincrement': increment_subincrement, '32-digit-key': get_32_digit_key, 'first_key-first_value-second_key-second_value': first_key_first_value_second_key_second_value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hatebase-hatebase.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

