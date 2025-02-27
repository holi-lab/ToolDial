import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sources(country: str=None, language: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "News sources endpoint provides access to the subsets of news publishers that the latest news is available from.
		
		It's mainly a convenience endpoint that you can use to keep track of the publishers available on the API, and you can pipe it straight through to your users"
    country: Find sources that display news in a specific country. Possible Options: us gb in jp ae sa au ca sg
        language: Find sources that display news in a specific language. Possible Options: en ar jp in es fr
        category: Find sources that display news of this category. Possible Options: top business science technology sports health entertainment
        
    """
    url = f"https://newsdata2.p.rapidapi.com/sources"
    querystring = {}
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newsdata2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(language: str=None, country: str=None, page: int=None, category: str=None, qintitle: str=None, q: str=None, domain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The latest news endpoint provides access to the latest and breaking news for a country, for a specific category in a country, or for a single or multiple domains. The news articles are sorted by the publish date.
		
		Retrieving the latest news allows you to build experience such as showcasing the latest news, breaking news tickers and analyzing News to better understand their content."
    language: A comma separated string of languages (maximum 5) to restrict the search to. Possible Options: en ar jp in es fr
        country: A comma separated string of 2-letter ISO 3166-1 countries (maximum 5) to restrict the search to. Possible Options: us gb in jp ae sa au ca
        page: Use this to page through the results if the total results found is greater than the page size.


        category: A comma separated string of categories (maximum 5) to restrict the search to. Possible Options: top business science technology sports health entertainment
        qintitle: Keywords or phrases to search for in the news title only.

Please note: This parameter can't be used with q parameter in the same query.
        q: Keywords or phrases to search for in the news title and content. 
Please note: You can't use AND and OR in the same query.
        domain: A comma separated string of domains (maximum 5) to restrict the search to. Use the /domains endpoint to find top sources id.
        
    """
    url = f"https://newsdata2.p.rapidapi.com/news"
    querystring = {}
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if category:
        querystring['category'] = category
    if qintitle:
        querystring['qintitle'] = qintitle
    if q:
        querystring['q'] = q
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newsdata2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def archive(domain: str=None, q: str=None, qintitle: str=None, category: str=None, from_date: str=None, language: str=None, page: int=None, to_date: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The news archive endpoint provides access to the old news data for a country, for a specific category in a country, or for a single or multiple domains.
		
		Retrieving a piece of old news allows you to get the past news data for research and analysis purposes."
    domain: A comma separated string of domains (maximum 5) to restrict the search to. Use the /domains endpoint to find top sources id.
        q: Keywords or phrases to search for in the news title and content. The value must be URL-encoded
Please note: You can't use AND and OR in the same query.
        qintitle: Keywords or phrases to search for in the news title only.

Please note: This parameter can't be used with q parameter in the same query.
        category: A comma separated string of categories (maximum 5) to restrict the search to. Possible Options: top business science technology sports health entertainment
        from_date: A date and optional time for the oldest article allowed. This should be in ISO 8601 format 
        language: A comma separated string of languages (maximum 5) to restrict the search to. Possible Options: en ar jp in es fr
        page: Use this to page through the results if the total results found is greater than the page size.

        to_date: A date and optional time for the newest article allowed. This should be in ISO 8601 format 
        country: A comma separated string of 2-letter ISO 3166-1 countries (maximum 5) to restrict the search to. Possible Options: us gb in jp ae sa au ca
        
    """
    url = f"https://newsdata2.p.rapidapi.com/archive"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    if q:
        querystring['q'] = q
    if qintitle:
        querystring['qintitle'] = qintitle
    if category:
        querystring['category'] = category
    if from_date:
        querystring['from_date'] = from_date
    if language:
        querystring['language'] = language
    if page:
        querystring['page'] = page
    if to_date:
        querystring['to_date'] = to_date
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newsdata2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

