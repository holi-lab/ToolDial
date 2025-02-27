import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_of_countries_with_income_level(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sourced from Developer Information issued by The World Bank Group at https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information
		
		External Doc URL available at http://api.worldbank.org/V2/incomeLevel/LIC/country
		
		Output format available at http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json"
    
    """
    url = f"https://world-bank-gdp.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-bank-gdp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_countries(iso2code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sourced from Developer Information issued by The World Bank Group at https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information
		
		External Doc URL available at http://api.worldbank.org/v2/country
		
		Output format available at http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json"
    
    """
    url = f"https://world-bank-gdp.p.rapidapi.com/"
    querystring = {'iso2Code': iso2code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-bank-gdp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

