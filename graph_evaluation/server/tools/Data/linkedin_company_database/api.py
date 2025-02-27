import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def linkedin_company_info(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint allows you to get data from LinkedIn Company Page."
    query: **PUBLIC LINKEDIN URL** of the target company (e.g `https://www.linkedin.com/company/microsoft`, `microsoft`). Hello Tom, we do not yet support LinkedIn Company URLs with the ID like this: https://www.linkedin.com/company/1073820. This will be available in the next 2-3 months. (updated 15 septembre 2023)
        
    """
    url = f"https://linkedin-company-database.p.rapidapi.com/v2/linkedin/company/info"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-company-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

