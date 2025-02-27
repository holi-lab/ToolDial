import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def meta_get_the_meta_data_about_surah_pages_hibz_and_juz(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the meta data about the Qur'an available in this API"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/meta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manzil_get_a_manzil_of_the_quran(edition: str, manzil: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Quran has 7 Manzils (for those who want to read / recite it over one week). You can get the text for each Manzil using the endpoint below.
		
		Returns the requested manzil from a particular edition
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/manzil/{manzil}/{edition}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ruku_get_a_ruku_of_the_quran(edition: str, ruku: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Quran has 556 Rukus. You can get the text for each Ruku using the endpoint below.
		
		Returns the requested ruku from a particular edition
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/ruku/{ruku}/{edition}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page_get_a_page_of_the_quran(page: str, edition: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the requested page from a particular edition
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/page/{page}/{edition}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hizb_quarter(hizb: str, edition: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Quran comprises 240 Hizb Quarters. One Hizb is half a Juz.
		
		Returns the requested Hizb Quarter from a particular edition
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/hizbQuarter/{hizb}/{edition}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sajda_get_all_verses_requiring_sajda(edition: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Depending on the madhab, there can be 14, 15 or 16 sajdas. This API has 15.
		Returns all the sajda ayahs from a particular edition
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/sajda/{edition}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ayah_get_an_ayah_of_the_quran(edition: str, reference: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Quran contains 6236 verses. With this endpoint, you can retrieve any of those verses.
		
		Returns an ayah for a given edition.
		{{reference}} here can be the ayah number or the surah:ayah. For instance, 262 or 2:255 will both get you Ayat Al Kursi
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/ayah/{reference}/{edition}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def juz_get_a_juz_of_the_quran(juz: str, edition: str, offset: str='3', limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the requested juz from a particular edition
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation
		
		Optional Parameters:
		offset - Offset ayahs in a juz by the given number
		limit - This is the number of ayahs that the response will be limited to."
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/juz/{juz}/{edition}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quran_get_a_complete_quran_edition(edition: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a complete Quran edition in the audio or text format
		{{edition}} is an edition identifier. Example: en.asad for Muhammad Asad's english translation"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/quran/{edition}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_edition_for_given_format(format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all editions for a given format
		{{format}} can be 'audio' or 'text'"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/edition/format/{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def edition_formats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all formats"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/edition/format"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_editions_for_type(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all editions for a given type
		{{type}} can be 'translation', 'tafsir' or another result returned in 4 above"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/edition/type/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def types_of_editions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all types of editions"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/edition/type"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_editions_for_language(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all editions for a given language
		{{language}} is a 2 digit language code. Example: en for English, fr for French, ar for Arabic"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/edition/language/{language}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all languages in which editions are available"
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/edition/language"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lists_all_available_editions(format: str='audio', language: str='en', type: str='versebyverse', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can filter the results using the parameters below.
		Parameters
		- format.... Specify a format. 'text' or 'audio
		- language.... A 2 digit language code. Example: 'en', 'fr', etc.
		- type... A valid type. Example - 'versebyverse', 'translation' etc."
    
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/edition"
    querystring = {}
    if format:
        querystring['format'] = format
    if language:
        querystring['language'] = language
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_surah(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "give digits in URL from 1 to 114 and get surah with different translations for now we are giving only two languages,
		but we are working on adding more soon.
		![](https://go4quiz.com/wp-content/uploads/List-of-114-Surahs-go4quiz.jpg)"
    id: 1 to 114
        is_id: 1 to 114
        
    """
    url = f"https://al-qur-an-all-translations.p.rapidapi.com/v1/surah/{is_id}"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "al-qur-an-all-translations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

