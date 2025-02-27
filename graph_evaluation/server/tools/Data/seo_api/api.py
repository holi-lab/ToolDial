import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Video Search
		
		Parameters
		----------
		query : A url encoded query string, for reference checkout https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters
		
		Returns
		-------
		json: a list of results with the link, description, and title for each result"
    query: A url encoded query string, for reference checkout https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters
        
    """
    url = f"https://seo-api.p.rapidapi.com/v1/video/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_b(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform SEO search for secondary search engine"
    
    """
    url = f"https://seo-api.p.rapidapi.com/v1/b/search/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Job Search
		
		Currently only supports jobs in North America
		
		Parameters
		----------
		query : Position Title and Location (optional). The query should be url encoded
		
		Returns
		-------
		json: a list of jobs with their link to apply, description, and title for each job opportunity"
    query: A url encoded query string, for reference checkout https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters
        
    """
    url = f"https://seo-api.p.rapidapi.com/v1/job/search/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def serp_get(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Google Search and search for website in Search Engine Results Pages (SERP)
		
		Parameters
		----------
		query : the string query to perform search. supports advance queries. Check out https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters guide for formatting, it is recommended to set the query `&num=100`
		
		Returns
		-------
		json: a list of results with the query, website, searched_results, and position. json["position"] will be set to -1 if website is not found in results"
    query: Required params `domain\" or `website` When using website please ensure that the website is url encoded
        
    """
    url = f"https://seo-api.p.rapidapi.com/v1/serp/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Google Image Search
		
		Parameters
		----------
		query : the string query to perform search. supports advance queries.
		
		Returns
		-------
		json: a list of results with the link, description, title, and image thumbnail for each result"
    
    """
    url = f"https://seo-api.p.rapidapi.com/v1/image/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crawl(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Google Search
		
		 Parameters
		----------
		query : the string query to perform search. supports advance queries. Check out https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters guide for formating
		
		Returns
		-------
		json: a the html source of the results page"
    
    """
    url = f"https://seo-api.p.rapidapi.com/v1/crawl/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Google Search
		
		Parameters
		----------
		query : A url encoded query string, for reference checkout https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters
		
		Returns
		-------
		json: a list of results with the link, description, and title for each result"
    query: A url encoded query string, for reference checkout https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters
        
    """
    url = f"https://seo-api.p.rapidapi.com/v1/search/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scholar(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Scholar Search
		
		Parameters
		----------
		query : A url encoded query string, for reference checkout https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters
		
		Returns
		-------
		json: a list of results with the link, description, and title for each result"
    query: A url encoded query string, for reference checkout https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters
        
    """
    url = f"https://seo-api.p.rapidapi.com/v1/scholar/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Google News Search
		
		 Parameters
		----------
		query : the string query to perform search for Google news. A simple query for `president` will be `q=president`. An example of multiple keyword would be `q=news+about+presdient+trump`. News can also be filtered by country and language. For `US` news and in English the query will be `q=trump&ceid=US:en` for news in Great Britian the query will be `q=trump&ceid=GB:en`
		
		Returns
		-------
		json: {"feed": { "title" : "trump" ...} , "entites": [ {"title" : "Trump doubles down on divisive messaging in speech to honor Independence Day - CNN", "links": []} ...]}"
    
    """
    url = f"https://seo-api.p.rapidapi.com/v1/news/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for products rankings"
    
    """
    url = f"https://seo-api.p.rapidapi.com/v1/product/search/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It "status" == true then API is up"
    
    """
    url = f"https://seo-api.p.rapidapi.com/v1/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

