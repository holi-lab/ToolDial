import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_emails(email_domain: str, query: str, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the web for emails using a query and email domain and get up to 5,000 email addresses."
    email_domain: Email domain - typically a company domain (e.g. wsgr.com) or an email provider domain (e.g. gmail.com).
        query: Search query.

`e.g.` *`steve smith san francisco ca usa`*
`e.g.` *`facebook ceo`*
`e.g.` *`jack blogger new york`*
`e.g.` *`car dealer california usa`*
        limit: Maximum number of emails to return. Accepts values from 1-5000.
**Default:** 5000
        
    """
    url = f"https://email-search16.p.rapidapi.com/search-emails"
    querystring = {'email_domain': email_domain, 'query': query, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-search16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

