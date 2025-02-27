import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_recipe(rid: str='37859', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    rid: rId: Id of desired recipe as returned by Search Query
        
    """
    url = f"https://community-food2fork.p.rapidapi.com/get"
    querystring = {}
    if rid:
        querystring['rId'] = rid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-food2fork.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str='shredded chicken', sort: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: q: (optional) Search Query (Ingredients should be separated by commas). If this is omitted top rated recipes will be returned.
        sort: sort: (optional) How the results should be sorted. The Food2Fork API offers two kinds of sorting for queries. The first is by rating. This rating is based off of social media scores to determine the best recipes. sort=r The second is by trendingness. The most recent recipes from our publishers have a trend score based on how quickly they are gaining popularity. sort=t
        page: page: (optional) Used to get additional results. Any request will return a maximum of 30 results. To get the next set of results send the same request again but with page = 2  The default if omitted is page = 1
        
    """
    url = f"https://community-food2fork.p.rapidapi.com/search"
    querystring = {}
    if q:
        querystring['q'] = q
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-food2fork.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

