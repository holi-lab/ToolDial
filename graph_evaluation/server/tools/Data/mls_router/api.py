import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_properties(orderby: str='ModificationTimestamp desc', select: str=None, top: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "top: this parameter determines the number of records on each page (max is 200, default is 10).
		
		skip: this parameter determines the page number (default is 0).
		
		select: this parameter determines the needed fields (default is all). note: more than 20 fields cannot be selected explicitly.
		
		filter: this determines the filtered criteria which are implemented by users. note: filters must be defined in Odata format.
		
		orderby: this parameter sorts results by the defined field (default is ListingKey). note: this parameter accepts “asc” and “desc” as an argument (default is “asc”)."
    
    """
    url = f"https://mls-router1.p.rapidapi.com/reso/odata/Property"
    querystring = {}
    if orderby:
        querystring['orderby'] = orderby
    if select:
        querystring['select'] = select
    if top:
        querystring['top'] = top
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mls-router1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_user_existence(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check User Existence."
    
    """
    url = f"https://mls-router1.p.rapidapi.com/reso/login/check-user-existance"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mls-router1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

