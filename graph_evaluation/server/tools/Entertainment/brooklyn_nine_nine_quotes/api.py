import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_quotes_random_from(episode: str=None, character: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If both query params are provided only character quotes are returned. Sample Request:
		            
		    GET api/quotes/random/from?character=Amy
		    {
		       "Message": "Quote returned.",
		       "Data": {
		            "Character": "Amy",
		            "Episode": "Paranoia",
		            "QuoteText": "..."
		        }       
		    }
		    
		    GET api/quotes/random/from?episode=AC/DC
		    {
		       "Message": "Quote returned.",
		       "Data": {
		            "Character": "Rosa",
		            "Episode": "AC/DC",
		            "QuoteText": "..."
		        }       
		    }
		    
		    GET api/quotes/random/from?episode=a
		    {
		       "Message": "Quote not found.",
		       "Data": null       
		    }"
    
    """
    url = f"https://brooklyn-nine-nine-quotes.p.rapidapi.com/api/v1/quotes/random/from"
    querystring = {}
    if episode:
        querystring['episode'] = episode
    if character:
        querystring['character'] = character
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brooklyn-nine-nine-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_quotes_all_from(character: str=None, pagesize: int=None, pagenumber: int=None, episode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Can navigate pages by setting param pageNumber.
		Page size limited by maximum of 50, can be set lower.
		If both query params are provided only character quotes are returned. Request Sample:
		
		    GET api/quotes/all/from?character=Jake
		    {   
		        "PageNumber": 1,
		        "PageSize": 2,
		        "Message": "Quote returned.",
		        "Data": [
		            {
		                "Character": "Jake",
		                "Episode": "Honeymoon",
		                "QuoteText": "..."
		            },
		            {
		                "Character": "Jake",
		                "Episode": "M.E. Time",
		                "QuoteText": "..."
		            }
		        ]
		    }
		    
		    GET api/quotes/all/from?character=a
		    {
		       "Message": "Quote not found.",
		       "Data": null       
		    }"
    
    """
    url = f"https://brooklyn-nine-nine-quotes.p.rapidapi.com/api/v1/quotes/all/from"
    querystring = {}
    if character:
        querystring['character'] = character
    if pagesize:
        querystring['PageSize'] = pagesize
    if pagenumber:
        querystring['PageNumber'] = pagenumber
    if episode:
        querystring['episode'] = episode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brooklyn-nine-nine-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_quotes_find(character: str=None, pagenumber: int=None, searchterm: str=None, pagesize: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Can navigate pages by setting param pageNumber.
		Page size limited by maximum of 50, can be set lower.
		Can search for term in isolation but also filter by character.
		Case insensitive. Request Sample:
		
		    GET api/quotes/find?searchTerm=pet
		    {   
		        "PageNumber": 1,
		        "PageSize": 2,
		        "Message": "Quote returned.",
		        "Data": [
		            {
		                "Character": "Amy",
		                "Episode": "Greg and Larry",
		                "QuoteText": "..."
		            },
		            {
		                "Character": "Scully",
		                "Episode": "The Honeypot",
		                "QuoteText": "..."
		            }
		        ]
		    }
		    
		    GET api/quotes/find?character=Amy&searchTerm=pet
		    {   
		        "PageNumber": 1,
		        "PageSize": 2,
		        "Message": "Quote returned.",
		        "Data": [
		            {
		                "Character": "Amy",
		                "Episode": "Greg and Larry",
		                "QuoteText": "..."
		            },
		            {
		                "Character": "Amy",
		                "Episode": "Chocolate Milk",
		                "QuoteText": "..."
		            }
		        ]
		    }
		    
		    GET api/quotes/find?character=1
		    {
		       "Message": "Quote not found.",
		       "Data": null       
		    }"
    
    """
    url = f"https://brooklyn-nine-nine-quotes.p.rapidapi.com/api/v1/quotes/find"
    querystring = {}
    if character:
        querystring['character'] = character
    if pagenumber:
        querystring['PageNumber'] = pagenumber
    if searchterm:
        querystring['searchTerm'] = searchterm
    if pagesize:
        querystring['PageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brooklyn-nine-nine-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_quotes_all(pagesize: int=None, pagenumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Can navigate pages by setting param pageNumber.
		Page size limited by maximum of 50, can be set lower. Sample Request:
		
		    GET api/quotes/all
		    {   
		        "PageNumber": 1,
		        "PageSize": 2,
		        "Message": "Quote returned.",
		        "Data": [
		            {
		                "Character": "Captain Holt",
		                "Episode": "Honeymoon",
		                "QuoteText": "..."
		            },
		            {
		                "Character": "Hitchcock",
		                "Episode": "M.E. Time",
		                "QuoteText": "..."
		            }
		        ]
		    }
		    
		    GET api/quotes/all?PageNumber=500
		    {
		       "Message": "Quote not found.",
		       "Data": null       
		    }"
    
    """
    url = f"https://brooklyn-nine-nine-quotes.p.rapidapi.com/api/v1/quotes/all"
    querystring = {}
    if pagesize:
        querystring['PageSize'] = pagesize
    if pagenumber:
        querystring['PageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brooklyn-nine-nine-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_quotes_random(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sample request:
		            
		    GET api/quotes/random
		    {
		       "Message": "Quote returned.",
		       "Data": {
		            "Character": "Charles",
		            "Episode": "Paranoia",
		            "QuoteText": "..."
		        }       
		    }"
    
    """
    url = f"https://brooklyn-nine-nine-quotes.p.rapidapi.com/api/v1/quotes/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brooklyn-nine-nine-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

