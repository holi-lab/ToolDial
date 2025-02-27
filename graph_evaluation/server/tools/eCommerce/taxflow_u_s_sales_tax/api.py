import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_sales_tax_by_state(state_name: str='California', state_abbr: str='CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the sales tax of the state name or abbreviation requested."
    
    """
    url = f"https://taxflow-u-s-sales-tax.p.rapidapi.com/api/tax/state"
    querystring = {}
    if state_name:
        querystring['state_name'] = state_name
    if state_abbr:
        querystring['state_abbr'] = state_abbr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxflow-u-s-sales-tax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sales_tax_by_zip_code(zip_code: int, state_abbr: str='CA', round: bool=None, state_name: str='California', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will take in a zip code and return the sales tax for the requested area combining together the city, county, and state tax rates."
    
    """
    url = f"https://taxflow-u-s-sales-tax.p.rapidapi.com/api/tax/zip"
    querystring = {'zip_code': zip_code, }
    if state_abbr:
        querystring['state_abbr'] = state_abbr
    if round:
        querystring['round'] = round
    if state_name:
        querystring['state_name'] = state_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taxflow-u-s-sales-tax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

