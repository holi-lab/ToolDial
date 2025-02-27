import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract_data(website: str, type: str, page: str='1', per_page: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts or pages from any Wordpress website.
		Example url format: https://www.playtoearn.online
		type: posts, pages, tags, categories, comments, taxonomies, media, users, plugins, types, statuses, settings, themes, search, block-types, blocks, block-directory
		
		Note that Wordpress websites that explicitly disabled API you cannot scrape but 98% of website are scrapeable."
    
    """
    url = f"https://scrape-extract-any-wordpress.p.rapidapi.com/wordpress-scraper"
    querystring = {'website': website, 'type': type, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrape-extract-any-wordpress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

