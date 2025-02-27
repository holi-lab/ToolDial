import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_paid_games(limit: str='50', category: str=None, region: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top paid games chart. Supports getting the chart for specific Google Play categories."
    limit: Limit the number of results.

**Allowed values:** `1-200`
**Default:** `50`

Note: requests with a limit value between 101-200 will be charged double (2 requests).
        category: Get the chart in a specific Google Play category (e.g. *GAME_RACING*).
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/top-paid-games"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category:
        querystring['category'] = category
    if region:
        querystring['region'] = region
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_free_apps(limit: int=None, language: str='en', category: str=None, region: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top free apps chart. Supports getting the chart for specific Google Play categories."
    limit: Limit the number of results.

**Allowed values:** `1-200`
**Default:** `50`

Note: requests with a limit value between 101-200 will be charged double (2 requests).
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        category: Get the chart in a specific Google Play category (e.g. *SOCIAL*).
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/top-free-apps"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_paid_apps(limit: int=50, category: str=None, region: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top paid apps chart. Supports getting the chart for specific Google Play categories."
    limit: Limit the number of results.

**Allowed values:** `1-200`
**Default:** `50`

Note: requests with a limit value between 101-200 will be charged double (2 requests).
        category: Get the chart in a specific Google Play category (e.g. *SOCIAL*).
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/top-paid-apps"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category:
        querystring['category'] = category
    if region:
        querystring['region'] = region
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_reviews(app_id: str, device: str=None, region: str='us', sort_by: str=None, rating: str=None, limit: int=10, cursor: str=None, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all app reviews."
    app_id: App Id of the app for which to get reviews.
        device: Only return reviews made on a specific device type.
Allowed values: `PHONE`, `TABLET`, `CHROMEBOOK`.
Default: `PHONE`.
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        sort_by: How to sort the reviews in the results.
Allowed values: `MOST_RELEVANT`, `NEWEST`, `RATING`.
Default: `MOST_RELEVANT`.
        rating: Only return reviews with a certain star rating.
Allowed values: `ANY`, `ONE_STAR`, `TWO_STARS`, `THREE_STARS`, `FOUR_STARS`, `FIVE_STARS`.
Default: `ANY`.
        limit: Maximum number of reviews in the results.
        cursor: Specify a cursor from the previous request to get the next set of results.
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/app-reviews"
    querystring = {'app_id': app_id, }
    if device:
        querystring['device'] = device
    if region:
        querystring['region'] = region
    if sort_by:
        querystring['sort_by'] = sort_by
    if rating:
        querystring['rating'] = rating
    if limit:
        querystring['limit'] = limit
    if cursor:
        querystring['cursor'] = cursor
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the full list of Google Play app categories. The returned categories can be used with the Charts endpoints (as the category parameter)."
    
    """
    url = f"https://store-apps.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_grossing_apps(limit: int=50, category: str=None, region: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top grossing apps chart. Supports getting the chart for specific Google Play categories."
    limit: Limit the number of results.

**Allowed values:** `1-200`
**Default:** `50`

Note: requests with a limit value between 101-200 will be charged double (2 requests).
        category: Get the chart in a specific Google Play category (e.g. *SOCIAL*).
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/top-grossing-apps"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category:
        querystring['category'] = category
    if region:
        querystring['region'] = region
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_details(language: str='en', region: str='us', app_id: str='com.google.android.apps.subscriptions.red', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full app details."
    language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        app_id: App Id. Batching of up to 100 App Ids is supported by separating multiple ids by comma (e.g. com.snapchat.android,com.microsoft.teams).
        
    """
    url = f"https://store-apps.p.rapidapi.com/app-details"
    querystring = {}
    if language:
        querystring['language'] = language
    if region:
        querystring['region'] = region
    if app_id:
        querystring['app_id'] = app_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, language: str='en', cursor: str=None, region: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for apps on the Store."
    q: Search query.
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        cursor: Specify a cursor from the previous request to get the next set of results.
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/search"
    querystring = {'q': q, }
    if language:
        querystring['language'] = language
    if cursor:
        querystring['cursor'] = cursor
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_grossing_games(limit: int=50, language: str='en', category: str=None, region: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top grossing games chart. Supports getting the chart for specific Google Play categories."
    limit: Limit the number of results.

**Allowed values:** `1-200`
**Default:** `50`

Note: requests with a limit value between 101-200 will be charged double (2 requests).
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        category: Get the chart in a specific Google Play category (e.g. *GAME_RACING*).
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/top-grossing-games"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_free_games(limit: int=50, category: str=None, language: str='en', region: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top free games chart. Supports getting the chart for specific Google Play categories."
    limit: Limit the number of results.

**Allowed values:** `1-200`
**Default:** `50`

Note: requests with a limit value between 101-200 will be charged double (2 requests).
        category: Get the chart in a specific Google Play category (e.g. *GAME_RACING*).
        language: The language to use, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        region: The country code of country/region to use, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default**: `us`.
        
    """
    url = f"https://store-apps.p.rapidapi.com/top-free-games"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category:
        querystring['category'] = category
    if language:
        querystring['language'] = language
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "store-apps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

