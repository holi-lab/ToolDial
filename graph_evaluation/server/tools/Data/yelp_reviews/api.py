import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def business_search(location: str, query: str, start: int=0, price_range: str=None, yelp_domain: str='yelp.com', sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Yelp businesses by query / keyword and location."
    location: Search location (e.g. *New York, NY, USA*).
        query: Search query / keyword.

**Examples:**
- *Alansi’s Plumbing*
- *Chinese restaurants*
        start: Number of results to skip. Mainly used for pagination.

**Default:** `0`.

**Allowed values:** positive integers.
        price_range: Find business with specific price ranges, specified as a comma delimited list of the following values: `$`, `$$`, `$$$`, `$$$`.

**Examples:**
- *$$*
- *$,$$*
        yelp_domain: The yelp domain to use for the search.

**Default:** `yelp.com`

**Supported domains:** `yelp.com`, `yelp.com.au`, `yelp.co.nz`, `ms.yelp.my`, `yelp.cz`, `yelp.dk`, `yelp.de`, `yelp.at`, `de.yelp.ch`, `en.yelp.be`, `yelp.ca`, `en.yelp.com.hk`, `en.yelp.my`, `en.yelp.com.ph`, `yelp.ie`, `yelp.com.sg`, `en.yelp.ch`, `yelp.co.uk`, `yelp.com.ar`, `yelp.cl`, `yelp.es`, `yelp.com.mx`, `fil.yelp.com.ph`, `yelp.fr`, `fr.yelp.ca`, `fr.yelp.ch`, `fr.yelp.be`, `yelp.no`, `yelp.pl`, `yelp.pot`, `yelp.com.br`, `fi.yelp.fi`, `sv.yelp.fi`, `yelp.se`, `yelp.com.tr`, `yelp.co.jp`, `zh.yelp.com.hk`, `yelp.com.tw`.
        sort_by: Return the results in a specific sort order.

**Default:** `RECOMMENDED`

**Allowed values:** `RECOMMENDED, HIGHEST_RATED, REVIEW_COUNT`
        
    """
    url = f"https://yelp-reviews.p.rapidapi.com/business-search"
    querystring = {'location': location, 'query': query, }
    if start:
        querystring['start'] = start
    if price_range:
        querystring['price_range'] = price_range
    if yelp_domain:
        querystring['yelp_domain'] = yelp_domain
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yelp-reviews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_reviews(business_id: str, num_pages: str='1', query: str='cheese', sort: str=None, language: str='en', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a single or multiple business review pages from Yelp (each page includes up to 10 reviews)."
    business_id: Yelp business id or alias to fetch reviews for.

**Examples**
- *WHJ2spR-_1P_tbiOqOibjg*
- *pearls-deluxe-burgers-san-francisco-3*
        num_pages: Number of pages to return, starting from `page`.

**Default:** `1`.

**Allowed values:** 1-10.

 **Note**: requests for more than one page are charged 2x.
        query: Return reviews matching a text query.
        sort: How to sort reviews in the results. Note that the `ELITES` value will cause the API to only return reviews posted by elite members.

**Default:** `BEST_MATCH`

**Allowed values:** `BEST_MATCH`, `NEWEST`, `OLDEST`, `HIGHEST_RATED`, `LOWEST_RATED`, `ELITES`.
        language: Only return reviews of a certain language. For a list of supported language codes see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 .

**Default:** `en`.
        page: Page to return (each page includes up to 10 results).

**Default:** `1`.
        
    """
    url = f"https://yelp-reviews.p.rapidapi.com/business-reviews"
    querystring = {'business_id': business_id, }
    if num_pages:
        querystring['num_pages'] = num_pages
    if query:
        querystring['query'] = query
    if sort:
        querystring['sort'] = sort
    if language:
        querystring['language'] = language
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yelp-reviews.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

