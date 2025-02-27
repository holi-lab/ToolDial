import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ww_stats_record_id(record_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info one entry"
    record_id: ID of the record
        
    """
    url = f"https://covid-19-info.p.rapidapi.com/ww_stats/{record_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ww_stats(pretty: bool=None, max_results: int=1, sort: str='_id', where: str='countries_and_territories==Russia', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full info"
    pretty: Pretty formatted response for CLI for example.
        max_results: Max results on response
        sort: Sorting flag, should be a field name. For reverse sort just use `-` in parameter name: `sort=field1, -field2, field3`
        where: MongoDB standard selector (details: https://docs.mongodb.com/manual/reference/method/db.collection.find/). Also able to use standard python operations like `==`/`>`/`<`/`!=`
        page: Resonse page, that parameter used for implementation of pagination
        
    """
    url = f"https://covid-19-info.p.rapidapi.com/ww_stats"
    querystring = {}
    if pretty:
        querystring['pretty'] = pretty
    if max_results:
        querystring['max_results'] = max_results
    if sort:
        querystring['sort'] = sort
    if where:
        querystring['where'] = where
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

