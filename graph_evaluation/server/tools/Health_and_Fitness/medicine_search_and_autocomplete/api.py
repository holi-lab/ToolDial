import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_medicine(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows you to retrieve detailed information about a specific medicine based on its ID. Simply provide the ID in the URL parameter, and you'll receive a response with information such as the medicine's name, price, content, and the company that produces it."
    
    """
    url = f"https://medicine-search-and-autocomplete.p.rapidapi.com/api/medicine/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medicine-search-and-autocomplete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def medicine_search(searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Medicine Search API **
		
		This API endpoint provides an autocomplete functionality that allows users to search for medicines by their name, content, or company name. The endpoint takes a search term as a query parameter and returns a list of medicines that match the term, along with relevant information such as the medicine's ID, name, price, content, company name, and a ranking score based on how closely the medicine matches the search term. This endpoint is useful for building an autocomplete feature in your application that can help users quickly find the medicines they need"
    
    """
    url = f"https://medicine-search-and-autocomplete.p.rapidapi.com/api/medicine/search"
    querystring = {'searchterm': searchterm, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "medicine-search-and-autocomplete.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

