import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scrape_contact_details_from_a_website(url: str, exclude_emails: str='contact@yellowpagesvn.com', exclude_phones: str='+84. 24 3636 9512,+84) 914 261 828,+84. 24 3636 9371', only_valid_phone_numbers: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint will scrape emails and phones from any website using it`s URL"
    url: the url of the website
        exclude_emails: **If you want to scrape a website that displays many contacts or many profiles but there is an email or an email of the website appearing and you don't need it, The API won't return any of these emails if discovered on the website.**
***you can also input several emails separated by commas.***
        exclude_phones: **The API is helpful if you want to scrape a website that shows many contacts and many profiles but there is a phone number or the phone number of the website displayed on the page and you don't need it. You can enter many numbers separated by a comma. The API won't return any of these phone numbers if found on the website.**

### Please don't forget to quote the parameter if there are symbols like this

```
# before  +84. 24 3636 9512,+84) 914 261 828,+84. 24 3636 9371
unwanted_phones=\\\\\\\"%2B84. 24 3636 9512,%2B84) 914 261 828,%2B84. 24 3636 9371\\\\\\\"

```
        only_valid_phone_numbers: true means: check the phones numbers and return only those looks real
support two values
true and false
        
    """
    url = f"https://contact-scraper.p.rapidapi.com/contact_scraper"
    querystring = {'url': url, }
    if exclude_emails:
        querystring['exclude_emails'] = exclude_emails
    if exclude_phones:
        querystring['exclude_phones'] = exclude_phones
    if only_valid_phone_numbers:
        querystring['only_valid_phone_numbers'] = only_valid_phone_numbers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contact-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

