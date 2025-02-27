import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keyword_suggestions_for_site(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of suggested keywords for a given website. The response includes the keyword text, competition level, competition index, search volume, and top page bid range for each keyword.. These data points can help you understand the relative difficulty and value of ranking for each keyword on a specific website."
    url: A string containing the URL of the website you want to get keyword suggestions for.
        
    """
    url = f"https://rankonyx.p.rapidapi.com/keyword/suggestions/site"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankonyx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyword_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of keyword categories in various industries. The categories are organized by industry, and each category includes a list of related keywords. This data can help you understand the types of keywords that are commonly used within each industry."
    
    """
    url = f"https://rankonyx.p.rapidapi.com/keyword/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankonyx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_locations_by_query(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API allows you to search for locations by name and retrieve detailed information about them. You can use the API to get information such as the location's name, unique 4-digit code, type (either "State" or "Country"), and ISO 3166-1 alpha-2 country code. The location code can be used to query SEO keywords within the specified location."
    
    """
    url = f"https://rankonyx.p.rapidapi.com/locations"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rankonyx.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

