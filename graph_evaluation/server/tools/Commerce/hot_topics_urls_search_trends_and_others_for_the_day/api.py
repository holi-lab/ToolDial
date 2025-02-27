import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hot_topics_of_today(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access this API to get the Hot Topics of Today. Everyday this API gives the hot topic for that day."
    
    """
    url = f"https://hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com/hotTopicsofToday"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_urls_of_today(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access this API to get the popular URLs of today.
		Everyday this API gives the popular URLs  for that day."
    
    """
    url = f"https://hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com/popularURLsofToday"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_the_trending_google_search_terms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find the trending Google search terms. This end point is applicable to be used everyday."
    
    """
    url = f"https://hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com/trendingGoogleSearches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_suggestion_on_google_search_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access this API to ask google to suggest you  the possible search categories. This end point is applicable to be used everyday."
    
    """
    url = f"https://hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com/suggestGoogleSearchCategories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hot-topics-urls-search-trends-and-others-for-the-day.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

