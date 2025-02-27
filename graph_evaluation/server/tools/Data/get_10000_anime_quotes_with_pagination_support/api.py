import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recent_quotes_by_pagination(page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent anime quotes without any categorization:
		Required GET parameter
		**page**:  1
		On the above **page**, 1 means it will fetch the latest 10 quotes if the page value is 2 then It will bring the next 10 latest quotes
		page 1: 0-10 
		page 2: 10-20 ......
		
		**RESPONSE**
		**quote** 
		Contain quote text
		
		**animename** 
		Japanese anime name, quotes related to.
		
		**character**  ( Depend on subscription )
		Character name who spoke that quote.
		
		**is_popular** ( Depend on subscription )
		tells whether a quote is popular among fans.
		Response will be either  1 or 0 ( 1 represent yes, 0 represent no )
		
		**quote_id** ( Depend on subscription )
		Unique quote id which can be later used to get more information.
		
		**image** (Depend on subscription)
		Character Image URL will be provided which is related to the quote.
		
		**Note: if no quote found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://10000-anime-quotes-with-pagination-support.p.rapidapi.com/rapidHandler/recent"
    querystring = {'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10000-anime-quotes-with-pagination-support.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_quote_based_on_quote_id(quote_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full quote info based on quote_id.
		
		Required GET parameter
		**quote_id**:  3702
		
		**RESPONSE**
		**quote** 
		Contain quote text
		
		**animename** 
		Japanese anime name, quotes related to.
		
		**character**  ( Depend on subscription )
		Character name who spoke that quote.
		
		**is_popular** ( Depend on subscription )
		tells whether a quote is popular among fans.
		Response will be either  1 or 0 ( 1 represent yes, 0 represent no )
		
		**quote_id** ( Depend on subscription )
		Unique quote id which can be later used to get more information.
		
		**image** (Depend on subscription)
		Character Image URL will be provided which is related to the quote.
		
		**Note: if no quote found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://10000-anime-quotes-with-pagination-support.p.rapidapi.com/rapidHandler/search_by_id"
    querystring = {'quote_id': quote_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10000-anime-quotes-with-pagination-support.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_quote_by_anime_name(search: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime quotes based on anime search
		
		Required GET parameter
		**page**:  1
		**search**:  Naruto
		On the above **page**, 1 means it will fetch the latest 10 quotes if the page value is 2 then It will bring the next 10 latest quotes
		page 1: 0-10 
		page 2: 10-20 ......
		
		On the above **search** means it will try to fetch all quotes related to that anime
		**Note:** Search param should contain atleast 3 characters
		
		**RESPONSE**
		**quote** 
		Contain quote text
		
		**animename** 
		Japanese anime name, quotes related to.
		
		**character**  ( Depend on subscription )
		Character name who spoke that quote.
		
		**is_popular** ( Depend on subscription )
		tells whether a quote is popular among fans.
		Response will be either  1 or 0 ( 1 represent yes, 0 represent no )
		
		**quote_id** ( Depend on subscription )
		Unique quote id which can be later used to get more information.
		
		**image** (Depend on subscription)
		Character Image URL will be provided which is related to the quote.
		
		**Note: if no quote found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://10000-anime-quotes-with-pagination-support.p.rapidapi.com/rapidHandler/search_by_anime"
    querystring = {'search': search, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10000-anime-quotes-with-pagination-support.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_anime_quote(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**RESPONSE**
		**quote** 
		Contain quote text
		
		**animename** 
		Japanese anime name, quotes related to.
		
		**character**  ( Depend on subscription )
		Character name who spoke that quote.
		
		**is_popular** ( Depend on subscription )
		tells whether a quote is popular among fans.
		Response will be either  1 or 0 ( 1 represent yes, 0 represent no )
		
		**quote_id** ( Depend on subscription )
		Unique quote id which can be later used to get more information.
		
		**image** (Depend on subscription)
		Character Image URL will be provided which is related to the quote.
		
		**Note: if no quote found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://10000-anime-quotes-with-pagination-support.p.rapidapi.com/rapidHandler/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10000-anime-quotes-with-pagination-support.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_of_the_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quote of the day.
		
		**RESPONSE**
		**quote** 
		Contain quote text
		
		**animename** 
		Japanese anime name, quotes related to.
		
		**character**  ( Depend on subscription )
		Character name who spoke that quote.
		
		**is_popular** ( Depend on subscription )
		tells whether a quote is popular among fans.
		Response will be either  1 or 0 ( 1 represent yes, 0 represent no )
		
		**quote_id** ( Depend on subscription )
		Unique quote id which can be later used to get more information.
		
		**image** (Depend on subscription)
		Character Image URL will be provided which is related to the quote.
		
		**Note: if no quote found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://10000-anime-quotes-with-pagination-support.p.rapidapi.com/rapidHandler/quote_of_the_day"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10000-anime-quotes-with-pagination-support.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_quote_by_character_name(search: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime quotes based on character search:
		Required GET parameter
		**page**:  1
		**search**: Lelouch
		
		On the above **page**, 1 means it will fetch the latest 10 quotes if the page value is 2 then It will bring the next 10 latest quotes
		page 1: 0-10 
		page 2: 10-20 ......
		
		On the above **search** means it will try to fetch all quotes spoken by that character
		**Note:**  Search param should contain atleast 3 characters.
		
		**RESPONSE**
		**quote** 
		Contain quote text
		
		**animename** 
		Japanese anime name, quotes related to.
		
		**character**  ( Depend on subscription )
		Character name who spoke that quote.
		
		**is_popular** ( Depend on subscription )
		tells whether a quote is popular among fans.
		Response will be either  1 or 0 ( 1 represent yes, 0 represent no )
		
		**quote_id** ( Depend on subscription )
		Unique quote id which can be later used to get more information.
		
		**image** (Depend on subscription)
		Character Image URL will be provided which is related to the quote.
		
		**Note: if no quote found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://10000-anime-quotes-with-pagination-support.p.rapidapi.com/rapidHandler/search_by_character"
    querystring = {'search': search, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10000-anime-quotes-with-pagination-support.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_quotes_by_pagination(page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get anime quotes popular among fans:
		Required GET parameter
		**page**:  1
		On the above **page**, 1 means it will fetch the latest 10 quotes if the page value is 2 then It will bring the next 10 latest quotes
		page 1: 0-10 
		page 2: 10-20 ......
		
		**RESPONSE**
		**quote** 
		Contain quote text
		
		**animename** 
		Japanese anime name, quotes related to.
		
		**character**  ( Depend on subscription )
		Character name who spoke that quote.
		
		**is_popular** ( Depend on subscription )
		tells whether a quote is popular among fans.
		Response will be either  1 or 0 ( 1 represent yes, 0 represent no )
		
		**quote_id** ( Depend on subscription )
		Unique quote id which can be later used to get more information.
		
		**image** (Depend on subscription)
		Character Image URL will be provided which is related to the quote.
		
		**Note: if no quote found response will be**
		`{"status": "empty"}`"
    
    """
    url = f"https://10000-anime-quotes-with-pagination-support.p.rapidapi.com/rapidHandler/popular"
    querystring = {'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "10000-anime-quotes-with-pagination-support.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

