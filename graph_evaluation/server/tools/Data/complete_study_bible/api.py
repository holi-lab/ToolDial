import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_strongs_number_api(include_lxx: bool, strongs_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns every verse containing the supplied strong's number.
		Include lxx boolean option will search the Septuagint translation of the Old Testament when searching for a Greek word. This allows connections to be made between New Testament words and Old Testament concepts."
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/search-strongs/{strongs_number}/{include_lxx}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def strongs_number_detail_api(comma_separated_strongs_numbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receives a Strongs Number or comma-separated list of Strong's Numbers (limit 3 at a time) and returns detailed information for each number, including Greek and Hebrew, phonetic pronunciation guide, root, Strong's Condordance, Mounce's Dictionary, Thayer's Greek Lexicon, and Brown-Driver-Brigg's (BDB) Lexicon (Hebrew only).
		
		Thayer's and BDB are stored as html and should be displayed as raw html in the browser. Add CSS styling as desired.
		All scripture references in the html are stored as <a> tags with the following href format: "/Bible/{book}/{chapter}/{verse}". If you want the links to work properly, you can program your front-end to recognize these links and do an API call to the Chapter and Verse api  and display the response in a pop-up window."
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/strongs-detail/{comma_separated_strongs_numbers}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def full_chapter_api(chapter: int, translation: str, book: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a specified chapter of a book in the Bible.
		Book parameter will accept a full book name, or 3 letter book abbreviation (example: '1Co' = 1 Corinthians). For a full list of books, abbreviations, and chapter counts, send a request to the "Books" api.
		
		Translation options inlcude: "KJV", "KJV-Strongs", "ORIG", and "LXX".
		ORIG will return Hebrew with Strong's Numbers if book is in the Old Testament, Greek with Strong's Numbers if New Testament.
		LXX is Old Testament only, and returns the Apostolic Polyglot Bible - which is an English translation of the Greek Septuagint with underlying Strong's numbers.
		Translations with Strong's numbers will return each verse as a JSON array of objects like so:
		 [{"phrase": "In the beginning", "data_nums": ["G746"]}
		Some phrases are translated from multiple Strong's numbers in tandem, so the data_nums property is an array. You can display the Strong's numbers however you'd like, or hide them from view. You can get data on each Strong's number upon click by calling the "Strong's Detail API".
		
		"Places" represents an array of objects for each verse containing any known locations mentioned in the verse. Places provide lat / long information and can be displayed on a map using any mapping packages such as LeafletJS, MapboxGL, Google, etc. Detailed info on each place can be obtained by calling the ID number in the "Place Detail API". NOTE: some places have numbers in them... i.e. "Bethany 2". When this occurs, it is used for distinction and indicates that there is more than one place of the same name referenced in the Bible"
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/{book}/{chapter}/{translation}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_detail_api(place_unique_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used in conjunction with Chapter and Verse API. Whenever a city/region/location is mentioned in a verse, the Chapter and Verse API provides a unique ID for the location that is mentioned. This unique ID is more reliable than searching for a place "name", because place names and spellings can change, especially from OT to NT. By using the Place ID's provided with the Chapter and Verse API, you can call the Place API to fetch more detail on each place, which returns a name, location, and data from the Encyclopedia of the Bible (when available).
		
		Lat / Longs are included with each place as well, so if you want to use a mapping API (leafletJS, MapBox, etc), you can display real-world locations for every place.
		
		Encyclopedia of the Bible data is provided in html form, display as raw html and apply CSS as desired"
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/place/{place_unique_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verse_range_api(translation: str, verse_end: str, chapter: str, verse_start: str, book: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use when you only want to fetch a passage subset, not the entire chapter."
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/{book}/{chapter}/{verse_start}/{verse_end}/{translation}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def passage_of_the_day_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Responds with a passage of the day, KJV translation. Can be a single verse or multiple verses. Passages are pre-selected by the API owner and updated every 24 hours."
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/daily-passage/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_place_id_api(place_unique_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used in conjuction with Chapter and Verse API, Place Detail API. Receives a Place Unique ID, returns a list of all verses in the Bible that mention the place. Helpful for studying the Biblical history of a particular place."
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/search-place/{place_unique_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_exact_phrase_api(phrase: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches the KJV for the exact phrase supplied"
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/search-exact/{phrase}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_all_words_api(words: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns KJV verses containing all of the supplied words, in any order. Words are separated by '%20' i.e., spaces. Words are not case-sensitive."
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/search/{words}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def book_list_and_chapter_counts_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all books of the Bible, abbreviations, and chapter counts. Can be used as a reference for a chapter selection screen"
    
    """
    url = f"https://complete-study-bible.p.rapidapi.com/books/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-study-bible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

