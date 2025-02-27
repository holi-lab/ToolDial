import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetches_articles_from_a_specific_newspaper_based_on_the_provided_search_query(newspaperid: str, searchtext: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /news/newspaper/:newspaperId/:searchText
		Fetches articles from a specific newspaper based on the provided search query.
		
		URL parameters:
		
		newspaperId (string): The name of the newspaper.
		searchText (string): The search query to filter articles.
		Response: A JSON array of articles containing the title, url, and source of each article.
		
		URL parameters:
		
		searchText (string): The search query to filter articles.
		Response: A JSON array of articles containing the title, url, and source of each article."
    
    """
    url = f"https://websites-scrapper.p.rapidapi.com/news/newspaper/{newspaperid}/{searchtext}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "websites-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetches_articles_from_all_available_newspapers_based_on_the_provided_search_query(searchtext: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /news/:searchText
		Fetches articles from all available newspapers based on the provided search query.
		
		URL parameters:
		
		searchText (string): The search query to filter articles.
		Response: A JSON array of articles containing the title, url, and source of each article."
    
    """
    url = f"https://websites-scrapper.p.rapidapi.com/news/{searchtext}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "websites-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

