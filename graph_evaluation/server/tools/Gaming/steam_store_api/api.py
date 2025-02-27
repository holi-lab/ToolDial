import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def free_to_play(page: int, countrycode: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a full list of free to play games and other free products."
    page: Page numbering starts from 0, so the first page is 0.
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        language: To get available languages, just call the **Language list** endpoint from **General** section.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/items/free-to-play"
    querystring = {'page': page, 'countryCode': countrycode, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_items(orderby: str, showonlyspecialoffers: bool, language: str, countrycode: str, term: str, page: int, hidefreetoplay: bool, maxprice: int=None, tagids: str=None, supportedlang: str=None, os: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "👉 This is one of the most important endpoints. Retrieve data from the Steam search engine. The filters here have a powerful impact. With their use you can find anything that interests you. If you want to limit the games to a certain amount, read the description of maxPrice parameter below.
		
		Maximum number of results served from one call is 50. But very often the number of results is lower because we skip game packs (bundles) in the search engine. If you're interested in this, let us know."
    language: To get available languages, just call the **Language list** endpoint from **General** section.
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        term: Search for items containing a given term/phrase/query
        page: Page numbering starts from 0, so the first page is 0.
        maxprice: Limit games to price. To get the allowed maximum price value for the selected currency go to the search results: https://store.steampowered.com/search/?term=witcher&ndl=1 and find the 'Narrow by Price' filter.
        tagids: You can narrow down the results to selected tags. Enter here, after a comma, the ids of the tags from the endpoint: **Tags** -> Tag list
        supportedlang: To get available languages, just call the **Language list** endpoint from **General** section. You can add multiple languages separated by commas, e.g. ***english,french,german**
        os: Allowed values: **empty, win, mac, linux***. You can provide several OS separated by comma, e.g. **win,mac**. Leave the field empty if you don't want to use this filter.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/search/items"
    querystring = {'orderBy': orderby, 'showOnlySpecialOffers': showonlyspecialoffers, 'language': language, 'countryCode': countrycode, 'term': term, 'page': page, 'hideFreeToPlay': hidefreetoplay, }
    if maxprice:
        querystring['maxPrice'] = maxprice
    if tagids:
        querystring['tagIds'] = tagids
    if supportedlang:
        querystring['supportedLang'] = supportedlang
    if os:
        querystring['os'] = os
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def achievement_list(appid: int, countrycode: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a full list of achievements with information on what needs to be done to get the achievement and what percentage of players have achieved it."
    appid: 292030 is 'The Witcher® 3: Wild Hunt' AppId
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        language: To get available languages, just call the **Language list** endpoint from **General** section.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/items/{appid}/achievements"
    querystring = {'countryCode': countrycode, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dlc_details(appid: int, language: str, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns details of the selected DLC."
    appid: 1233340 is 'The Witcher 3: Wild Hunt - Blood and Wine Soundtrack'
        language: To get available languages, just call the **Language list** endpoint from **General** section.
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/dlcs/{appid}"
    querystring = {'language': language, 'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_reviews(languagefilter: str, browsefilter: str, ratingfilter: str, countrycode: str, appid: str, language: str, page: int, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "💡 The endpoint returns a complete list of user reviews. Use filters to narrow down the results as desired. If you want to get reviews from page 2, you must first grab data from page 1 and save the **cursor** value from response. Then set in the new call  **page=2** and **cursor** from page 1."
    languagefilter: Additional filter for user reviews.  To get available languages, just call the **Language list** endpoint from **General** section. You can also use here **All** and **Default** values.
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        appid: 292030 is 'The Witcher® 3: Wild Hunt' AppId
        language: To get available languages, just call the **Language list** endpoint from **General** section.
        page: Page numbering starts from 0, so the first page is 0.
        cursor: The cursor is used to retrieve the next page of reviews. If you need to retrieve page 3, you must first retrieve pages 1 and 2 and use the returned cursor values sequentially. Or **simply query** page 1 and get the special link we have prepared for you in the response: **links -> more**. Use it in the API to fetch the next page.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/items/{appid}/user-reviews"
    querystring = {'languageFilter': languagefilter, 'browseFilter': browsefilter, 'ratingFilter': ratingfilter, 'countryCode': countrycode, 'language': language, 'page': page, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dlc_list(appid: int, page: int, countrycode: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns full DLC list for selected item."
    appid: 292030 is 'The Witcher® 3: Wild Hunt' AppId
        page: Page numbering starts from 0, so the first page is 0.
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        language: To get available languages, just call the **Language list** endpoint from **General** section.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/items/{appid}/dlcs"
    querystring = {'page': page, 'countryCode': countrycode, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shortened_item_details(countrycode: str, appid: int, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "🎮 It retrieves shortened game data. Check the endpoint worth since it may contain information more important to you than the ones in "Full item details"."
    countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        appid: 292030 is 'The Witcher® 3: Wild Hunt' AppId
        language: To get available languages, just call the **Language list** endpoint from **General** section.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/items/{appid}"
    querystring = {'countryCode': countrycode, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def items_by_tag(countrycode: str, tab: str, page: int, tagid: int, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "👉 This is one of the most important endpoints. It returns games and other products based on the provided **TagID** . You can find the **TagID** by calling the **Tag list** endpoint from the **Tags** section. You can also specify which tab you're interested in by setting the appropriate **tab** value."
    countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        page: Page numbering starts from 0, so the first page is 0.
        tagid: TagId **19** means **Action**
        language: To get available languages, just call the **Language list** endpoint from **General** section.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/items/tags/{tagid}"
    querystring = {'countryCode': countrycode, 'tab': tab, 'page': page, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_code_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this codes in other endpoints to grab data from the country of your interest. The country code is very **important** because it determines the **currency** in which prices will be displayed."
    
    """
    url = f"https://steam-store-api.p.rapidapi.com/country-codes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def full_item_details(language: str, countrycode: str, appid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "🎮 It retrieves full game data such as: prices, discount, detailed descriptions, PEGI/ESRB, DLCs, system requirements, screenshots and much more! It also returns API links to conveniently download details of: DLCs, reviews, achievements."
    language: To get available languages, just call the **Language list** endpoint from **General** section.
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        appid: 292030 is 'The Witcher® 3: Wild Hunt' AppId
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/items/{appid}"
    querystring = {'language': language, 'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a full list of available languages. Use them in other endpoints to retrieve data in the language you're interested in."
    
    """
    url = f"https://steam-store-api.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tag_list(language: str, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the full list of tags. Use the tag IDs to handle other endpoints. If you want to get all the games e.g. with **Action** tag, just:
		- retrieve the list
		- find the Action tag 
		- take its ID 
		- put it into **Items by tag** endpoint
		It's so simple!"
    language: To get available languages, just call the **Language list** endpoint from **General** section.
        countrycode: To get available country codes, just call the **Country code list** endpoint from **General** section.
        
    """
    url = f"https://steam-store-api.p.rapidapi.com/tags"
    querystring = {'language': language, 'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-store-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

