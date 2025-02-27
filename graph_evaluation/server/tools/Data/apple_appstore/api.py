import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_developer_information(devid: str, country: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Retrieves a list of applications by the give developer id. Options:
		
		-> devId: the AppStore dev Id of the developer, for example 284882218 for Facebook.
		country: the two letter country code to get the app details from. Defaults to us. Note this also affects the language of the data.
		-> lang: language code for the result text. Defaults to undefined, so country specific language should be used automatically."
    
    """
    url = f"https://apple-appstore.p.rapidapi.com/developer"
    querystring = {'devId': devid, }
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-appstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_apps(term: str, num: str='2', country: str=None, page: str=None, idsonly: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Retrieves a list of apps that results of searching by the given term. Options:
		
		-> term: the term to search for (required).
		-> num: the amount of elements to retrieve. Defaults to 50.
		-> page: page of results to retrieve. Defaults to to 1.
		-> country: the two letter country code to get the similar apps from. Defaults to us.
		-> lang: language code for the result text. Defaults to en-us.
		-> idsOnly: (optional, defaults to false): skip extra lookup request. Search results will contain array of application ids."
    
    """
    url = f"https://apple-appstore.p.rapidapi.com/search"
    querystring = {'term': term, }
    if num:
        querystring['num'] = num
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if idsonly:
        querystring['idsOnly'] = idsonly
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-appstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_apps_copy(term: str, num: str='2', country: str=None, page: str=None, lang: str=None, idsonly: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Retrieves a list of apps that results of searching by the given term. Options:
		
		-> term: the term to search for (required).
		-> num: the amount of elements to retrieve. Defaults to 50.
		-> page: page of results to retrieve. Defaults to to 1.
		-> country: the two letter country code to get the similar apps from. Defaults to us.
		-> lang: language code for the result text. Defaults to en-us.
		-> idsOnly: (optional, defaults to false): skip extra lookup request. Search results will contain array of application ids."
    
    """
    url = f"https://apple-appstore.p.rapidapi.com/search"
    querystring = {'term': term, }
    if num:
        querystring['num'] = num
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    if idsonly:
        querystring['idsOnly'] = idsonly
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-appstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_suggested_apps(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Given a string returns up to 50 suggestions to complete a search query term."
    
    """
    url = f"https://apple-appstore.p.rapidapi.com/suggest"
    querystring = {'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-appstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_app_privacy(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Retrieves the ratings for the app. Currently only for US App Store. Options:
		
		id: the AppStore "trackId" of the app, for example 553834731 for Candy Crush Saga."
    
    """
    url = f"https://apple-appstore.p.rapidapi.com/privacy"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-appstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apps_listing(num: str='2', lang: str=None, category: str=None, country: str=None, fulldetail: bool=None, collection: str='TOP_FREE_IOS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Retrieves a list of applications from one of the collections at Apple Store. Options
		
		
		collection: the collection to look up. Defaults to collection.TOP_FREE_IOS, available options are:
		* TOP_MAC
		* TOP_FREE_MAC
		* TOP_GROSSING_MAC
		* TOP_PAID_MAC
		* NEW_IOS
		* NEW_FREE_IOS
		* NEW_PAID_IOS
		* TOP_FREE_IOS
		* TOP_FREE_IPAD
		* TOP_GROSSING_IOS
		* TOP_GROSSING_IPAD
		* TOP_PAID_IOS
		* TOP_PAID_IPAD
		
		-> category: the category to look up. This is a number associated with the genre for the application. Defaults to no specific category. Available options are:
		*BOOKS
		*BUSINESS
		*CATALOGS
		*EDUCATION
		*ENTERTAINMENT
		*FINANCE
		*FOOD_AND_DRINK
		*GAMES
		*GAMES_ACTION
		*GAMES_ADVENTURE
		*GAMES_ARCADE
		*GAMES_BOARD
		*GAMES_CARD
		*GAMES_CASINO
		*GAMES_DICE
		*GAMES_EDUCATIONAL
		*GAMES_FAMILY
		*GAMES_MUSIC
		*GAMES_PUZZLE
		*GAMES_RACING
		*GAMES_ROLE_PLAYING
		*GAMES_SIMULATION
		*GAMES_SPORTS
		*GAMES_STRATEGY
		*GAMES_TRIVIA
		*GAMES_WORD
		*HEALTH_AND_FITNESS
		*LIFESTYLE
		*MAGAZINES_AND_NEWSPAPERS
		*MAGAZINES_ARTS
		*MAGAZINES_AUTOMOTIVE
		*MAGAZINES_WEDDINGS
		*MAGAZINES_BUSINESS
		*MAGAZINES_CHILDREN
		*MAGAZINES_COMPUTER
		*MAGAZINES_FOOD
		*MAGAZINES_CRAFTS
		*MAGAZINES_ELECTRONICS
		*MAGAZINES_ENTERTAINMENT
		*MAGAZINES_FASHION
		*MAGAZINES_HEALTH
		*MAGAZINES_HISTORY
		*MAGAZINES_HOME
		*MAGAZINES_LITERARY
		*MAGAZINES_MEN
		*MAGAZINES_MOVIES_AND_MUSIC
		*MAGAZINES_POLITICS
		*MAGAZINES_OUTDOORS
		*MAGAZINES_FAMILY
		*MAGAZINES_PETS
		*MAGAZINES_PROFESSIONAL
		*MAGAZINES_REGIONAL
		*MAGAZINES_SCIENCE
		*MAGAZINES_SPORTS
		*MAGAZINES_TEENS
		*MAGAZINES_TRAVEL
		*MAGAZINES_WOMEN
		*MEDICAL
		*MUSIC
		*NAVIGATION
		*NEWS
		*PHOTO_AND_VIDEO
		*PRODUCTIVITY
		*REFERENCE
		*SHOPPING
		*SOCIAL_NETWORKING
		*SPORTS
		*TRAVEL
		*UTILITIES
		*WEATHER
		
		-> country: the two letter country code to get the list from. Defaults to us.
		-> lang: language code for the result text. Defaults to undefined, so country specific language should be used automatically.
		-> num: the amount of elements to retrieve. Defaults to 50, maximum allowed is 200.
		fulldetail: If this is set to true, an extra request will be made to get extra attributes of the resulting applications (like those returned by the app method)."
    category: please read the caption for more details
        fulldetail: true or false
        collection: please read the caption for more details
        
    """
    url = f"https://apple-appstore.p.rapidapi.com/list"
    querystring = {}
    if num:
        querystring['num'] = num
    if lang:
        querystring['lang'] = lang
    if category:
        querystring['category'] = category
    if country:
        querystring['country'] = country
    if fulldetail:
        querystring['fulldetail'] = fulldetail
    if collection:
        querystring['collection'] = collection
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-appstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_full_details(is_id: str, lang: str=None, country: str=None, ratings: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint Retrieves the full detail of an application. Options:
		
		->i d: the AppStore "trackId" of the app, for example 553834731 for Candy Crush Saga. 
		
		-> country: the two letter country code to get the app details from. Defaults to us. Note this also affects the language of the data.
		
		-> lang: language code for the result text. Defaults to undefined, so country specific language should be used automatically.
		
		-> ratings: load additional ratings information like ratings number and histogram"
    
    """
    url = f"https://apple-appstore.p.rapidapi.com/appinfo"
    querystring = {'id': is_id, }
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    if ratings:
        querystring['ratings'] = ratings
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apple-appstore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

