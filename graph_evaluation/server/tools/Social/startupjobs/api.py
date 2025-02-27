import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def job_offers(filters: str='isRemote eq 1', fields: str='name,description', limit: str='20', order: str='isRemote desc,name', offset: str='12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all job offers. Job offers are refreshing every 5 minutes."
    filters: A query to filter results.

Following operations are supported:
![](https://api.kub3c19.dev/public/img/operations.png?v=1)
        fields: A comma separated list of fields that are to be returned in results.
        limit: The number of results to return. If not specified, the limit is 10.
        order: An ordering query. For example:

order=isRemote desc,name

Above call will return results ordered by name in ascending order and then by isRemote in descending order.
        offset: Position to start fetching results from. Used for pagination.
        
    """
    url = f"https://startupjobs1.p.rapidapi.com/job-offers"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "startupjobs1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

