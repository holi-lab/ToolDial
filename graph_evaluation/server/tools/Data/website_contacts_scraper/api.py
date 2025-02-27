import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scrape_contacts_from_website(query: str, match_email_domain: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract emails, phone numbers and social profiles from website root domain domain."
    query: Domain from which to scrape emails and contacts (e.g. wsgr.com). Accepts any valid url and uses its root domain as a starting point for the extraction.
        match_email_domain: Only return emails in the same domain like the one supplied with the *query* parameter.
        
    """
    url = f"https://website-contacts-scraper.p.rapidapi.com/scrape-contacts"
    querystring = {'query': query, }
    if match_email_domain:
        querystring['match_email_domain'] = match_email_domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "website-contacts-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

