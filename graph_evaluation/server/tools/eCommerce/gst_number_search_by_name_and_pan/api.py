import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_details_by_director_name(term: str='belinda', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Director and their company details by director name"
    
    """
    url = f"https://gst-number-search-by-name-and-pan.p.rapidapi.com/director/autocomplete"
    querystring = {}
    if term:
        querystring['term'] = term
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gst-number-search-by-name-and-pan.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_details_by_company_name(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Suggestions To do GST Number search by name:-
		Enter the correct word (check spelling) with
		GST Verification
		
		- Enter at least ten characters of the word you want
		- Enter a state name because the company may have GST registration in various states.
		
		For example, if you want to do the GST number search by the name of an enterprise named GIRIRAJ TRADERS IN DELHI, then search with the name mentioned above. You will get all the registered information associated with GIRIRAJ TRADERS IN DELHI."
    
    """
    url = f"https://gst-number-search-by-name-and-pan.p.rapidapi.com/company/autocomplete"
    querystring = {'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gst-number-search-by-name-and-pan.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

