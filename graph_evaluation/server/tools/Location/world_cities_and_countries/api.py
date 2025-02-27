import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchcities(limit: int=100, page: int=1, countryisocode: str='US', keyword: str='new yo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API to search cities based on the matching search keyword passed in url param.
		e.g. 
		`/v1/locations/cities/new y`: This will return all the matching cities whose name starts from `new y`.
		You can also filter the cities or narrow down results for specific country by passing its 2 letter country code in query param which can be found from our searchCountry API."
    limit: This indicates how many results you would like to query in one time.
Default value is `10` if limit is not provided.
Maximum limit is `1000`
        page: This controls the pagination of results.
Default is `1`
        countryisocode: It is an optional parameter to narrow your search results.
Pass two letter Country code to  filter the cities for specific country in query param which can be found in our searchCountry API in field name `alpha-2` for any country you search.
for e.g. 'IN', 'US' etc.
        
    """
    url = f"https://world-cities-and-countries.p.rapidapi.com/v1/locations/cities/{keyword}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if countryisocode:
        querystring['countryIsoCode'] = countryisocode
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities-and-countries.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listallcountries(page: int=1, sortby: str='name:asc', limit: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns list of all countries and can be paginated over the results."
    sortby: Valid format to sort is `field:order`
e.g. `name:asc`, `name:desc` for country name sorting

where `asc` for sorting in ascending order
`desc` for sorting in descending order
        
    """
    url = f"https://world-cities-and-countries.p.rapidapi.com/v1/locations/countries"
    querystring = {}
    if page:
        querystring['page'] = page
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities-and-countries.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchcountries(keyword: str, page: int, limit: int, sortby: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API to search county based on the matching search keyword passed in url param.
		NOTE: This API is case insensitive.
		e.g. 
		`/v1/locations/countries/uni`: This will return all the matching countries whose name starts from `uni` and their additional info."
    sortby: Valid format to sort is `field:order`
e.g. `name:desc`, `alpha-2:asc`
where `asc` for sorting in ascending order
`desc` for sorting in descending order
        
    """
    url = f"https://world-cities-and-countries.p.rapidapi.com/v1/locations/countries/{keyword}"
    querystring = {'page': page, 'limit': limit, 'sortBy': sortby, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cities-and-countries.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

