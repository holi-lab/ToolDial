import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verses_by_page(page_number: int, fields: str=None, translations: str=None, language: str='en', audio: int=None, tafsirs: str=None, page: int=1, words: str='true', per_page: int=10, word_fields: str=None, translation_fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all verses of a specific Madani Mushaf page(1 to 604)"
    page_number: Madani Mushaf page number. Valid range is 1-604 
        fields: comma separated  list of ayah fields.
        translations: comma separated ids of translations to load for each ayah.
        language: Language to fetch word translation in specific language.
        audio: Id of recitation if you want to load audio of each ayah.
        tafsirs: Comma separated ids of tafisrs to load for each ayah if you want to load tafisrs.
        page: For paginating within the result
        words: Include words of each ayah?

0 or false will not include words.

1 or true will include the words.
        per_page: records per api call, you can get maximum 50 records. 
        word_fields: Comma separated list of word fields if you want to add more fields for each word. 
        translation_fields: Comma separated list of translation fields if you want to add more fields for each translation. 
        
    """
    url = f"https://quran-com.p.rapidapi.com/verses/by_page/{page_number}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if translations:
        querystring['translations'] = translations
    if language:
        querystring['language'] = language
    if audio:
        querystring['audio'] = audio
    if tafsirs:
        querystring['tafsirs'] = tafsirs
    if page:
        querystring['page'] = page
    if words:
        querystring['words'] = words
    if per_page:
        querystring['per_page'] = per_page
    if word_fields:
        querystring['word_fields'] = word_fields
    if translation_fields:
        querystring['translation_fields'] = translation_fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rub_el_hizb_recitations(rub_el_hizb_number: int, recitation_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of ayah recitations for a Rub el Hizb."
    recitation_id: Recitation Id, you can get list of all ayah by ayah recitations using this endpoint #endpoint:HLbauN2sdGitPQPPL
        
    """
    url = f"https://quran-com.p.rapidapi.com/recitations/{recitation_id}/by_rub/{rub_el_hizb_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uthmani_simple_text(verse_key: str=None, rub_el_hizb_number: int=None, chapter_number: int=None, page_number: int=None, juz_number: int=None, hizb_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Uthmani simple script(without tashkiq/diacritical marks) of ayah. Use query strings to filter results, leave all query string blank if you want to fetch script of whole Quran."
    verse_key: If you want to get Uthmani script of a specific ayah.
        rub_el_hizb_number: If you want to get Uthmani script of a specific Rub el Hizb.
        chapter_number: If you want to get Uthmani script of a specific surah.
        page_number: If you want to get Uthmani script of a Madani Muhsaf page
        juz_number: If you want to get Uthmani script of a specific juz.
        hizb_number: If you want to get Uthmani script of a specific hizb.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/verses/uthmani_simple"
    querystring = {}
    if verse_key:
        querystring['verse_key'] = verse_key
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if page_number:
        querystring['page_number'] = page_number
    if juz_number:
        querystring['juz_number'] = juz_number
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tafirs_by_id(tafsir_id: int, chapter_number: int=None, hizb_number: int=None, juz_number: int=None, page_number: int=None, rub_el_hizb_number: int=None, verse_key: str=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single tafsir
		
		See tafsirs endpoint to fetch available tafsirs.
		
		You can also include more fields of tafsir using `fields` query string."
    tafsir_id: tafsir id
        chapter_number: If you want to get tafsir of a specific surah.
        hizb_number: If you want to get tafsir of a specific hizb.
        juz_number: If you want to get tafsir of a specific juz.
        page_number: If you want to get tafsir of a Madani Muhsaf page
        rub_el_hizb_number: If you want to get tafsir of a specific Rub el Hizb.
        verse_key: If you want to get tafsir of a specific ayah.
        fields: comma seperated fields of tafsir.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/tafsirs/{tafsir_id}"
    querystring = {}
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if juz_number:
        querystring['juz_number'] = juz_number
    if page_number:
        querystring['page_number'] = page_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if verse_key:
        querystring['verse_key'] = verse_key
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def glyph_code_qcf_v1_font(verse_key: str=None, chapter_number: int=None, page_number: int=None, hizb_number: int=None, juz_number: int=None, rub_el_hizb_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get glyph codes of ayah for QCF v1 font"
    verse_key: If you want to get text of a specific ayah.
        chapter_number: If you want to get text of a specific surah.
        page_number: If you want to get text of a Madani Muhsaf page
        hizb_number: If you want to get text of a specific hizb.
        juz_number: If you want to get text of a specific juz.
        rub_el_hizb_number: If you want to get text of a specific Rub el Hizb.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/verses/code_v1"
    querystring = {}
    if verse_key:
        querystring['verse_key'] = verse_key
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if page_number:
        querystring['page_number'] = page_number
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if juz_number:
        querystring['juz_number'] = juz_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recitations_by_id(recitation_id: int, fields: str=None, rub_el_hizb_number: int=None, hizb_number: int=None, juz_number: int=None, verse_key: str=None, page_number: int=None, chapter_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of reciters for a single recitaiton. See reciters endpoint to fetch available recitations.
		
		You can also include more fields of audio files using `fields` query string."
    recitation_id: Recitation id
        fields: comma seperated field of audio files.
        rub_el_hizb_number: If you want to get audio file of a specific Rub el Hizb.
        hizb_number: If you want to get audio file of a specific hizb.
        juz_number: If you want to get audio file of a specific juz.
        verse_key: If you want to get audio file of a specific ayah.
        page_number: If you want to get audio file of a Madani Muhsaf page
        chapter_number: If you want to get audio file of a specific surah.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/recitations/{recitation_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if juz_number:
        querystring['juz_number'] = juz_number
    if verse_key:
        querystring['verse_key'] = verse_key
    if page_number:
        querystring['page_number'] = page_number
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indopak_text(chapter_number: int=None, hizb_number: int=None, juz_number: int=None, rub_el_hizb_number: int=None, page_number: int=None, verse_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Indopak script of ayah. Use query strings to filter results, leave all query string blank if you want to fetch Indopak script of whole Quran."
    chapter_number: If you want to get indopak script of a specific surah.
        hizb_number: If you want to get indopak script of a specific hizb.
        juz_number: If you want to get indopak script of a specific juz.
        rub_el_hizb_number: If you want to get indopak script of a specific Rub el Hizb.
        page_number: If you want to get indopak script of a Madani Muhsaf page
        verse_key: If you want to get indopak script of a specific ayah.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/verses/indopak"
    querystring = {}
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if juz_number:
        querystring['juz_number'] = juz_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if page_number:
        querystring['page_number'] = page_number
    if verse_key:
        querystring['verse_key'] = verse_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(size: int=20, q: str=None, language: str='en', page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search quran text"
    size: Results per page. *s* is also valid parameter.
        q: Search query, you can use *query* as well
        language: ISO code of language, use this query params if you want to boost translations for specific language.
For list of available language see  #endpoint:8rjMCZEyPEZMHAKz9 endpoint.
        page: Page number, well for pagination. You can use *p* as well
        
    """
    url = f"https://quran-com.p.rapidapi.com/search"
    querystring = {}
    if size:
        querystring['size'] = size
    if q:
        querystring['q'] = q
    if language:
        querystring['language'] = language
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tajweed_colored_uthmani_text(verse_key: str=None, hizb_number: int=None, chapter_number: int=None, juz_number: int=None, page_number: int=None, rub_el_hizb_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Uthmani color coded tajweed text of ayah. Tajweed rules are embeded in text as `tajweed` html tags."
    verse_key: If you want to get text of a specific ayah.
        hizb_number: If you want to get text of a specific hizb.
        chapter_number: If you want to get text of a specific surah.
        juz_number: If you want to get text of a specific juz.
        page_number: If you want to get text of a Madani Muhsaf page
        rub_el_hizb_number: If you want to get text of a specific Rub el Hizb.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/verses/uthmani_tajweed"
    querystring = {}
    if verse_key:
        querystring['verse_key'] = verse_key
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if juz_number:
        querystring['juz_number'] = juz_number
    if page_number:
        querystring['page_number'] = page_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reciters(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of reciters"
    language: Name of reciters in specific language. Will fallback to English if we don't have names in specific language.
        
    """
    url = f"https://quran-com.p.rapidapi.com/resources/chapter_reciters"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translation_by_id(translation_id: int, fields: str=None, hizb_number: int=None, page_number: int=None, verse_key: str=None, juz_number: int=None, rub_el_hizb_number: int=None, chapter_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single translation
		
		see translations endpoint to fetch available translations.
		
		You can also include more fields of translation using `fields` query string."
    translation_id: Recitation id
        fields: comma seperated fields of translation.
        hizb_number: If you want to get translation of a specific hizb.
        page_number: If you want to get translation of a Madani Muhsaf page
        verse_key: If you want to get translation of a specific ayah.
        juz_number: If you want to get translation of a specific juz.
        rub_el_hizb_number: If you want to get translation of a specific Rub el Hizb.
        chapter_number: If you want to get translation of a specific surah.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/translations/{translation_id}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if page_number:
        querystring['page_number'] = page_number
    if verse_key:
        querystring['verse_key'] = verse_key
    if juz_number:
        querystring['juz_number'] = juz_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verses_by_hizb(hizb_number: int, page: int=1, per_page: int=10, audio: int=None, translations: str=None, fields: str=None, word_fields: str=None, translation_fields: str=None, language: str='en', tafsirs: str=None, words: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all verses from a specific Hizb( half(1-60)."
    hizb_number: Hizb number(1-60)
        page: For paginating within the result
        per_page: records per api call, you can get maximum 50 records. 
        audio: Id of recitation if you want to load audio of each ayah.
        translations: comma separated ids of translations to load for each ayah.
        fields: comma separated  list of ayah fields.
        word_fields: Comma separated list of word fields if you want to add more fields for each word. 
        translation_fields: Comma separated list of translation fields if you want to add more fields for each translation. 
        language: Language to fetch word translation in specific language.
        tafsirs: Comma separated ids of tafisrs to load for each ayah if you want to load tafisrs.
        words: Include words of each ayah?

0 or false will not include words.

1 or true will include the words.
        
    """
    url = f"https://quran-com.p.rapidapi.com/verses/by_hizb/{hizb_number}"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if audio:
        querystring['audio'] = audio
    if translations:
        querystring['translations'] = translations
    if fields:
        querystring['fields'] = fields
    if word_fields:
        querystring['word_fields'] = word_fields
    if translation_fields:
        querystring['translation_fields'] = translation_fields
    if language:
        querystring['language'] = language
    if tafsirs:
        querystring['tafsirs'] = tafsirs
    if words:
        querystring['words'] = words
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chapters_audio(is_id: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of chapters audio for specific reciter"
    id: Id of reciter for which you want to get the recitations. You can fetch list of all reciters from this endpoint

#endpoint:ZQvDmxKn7fQwLrAfy
        
    """
    url = f"https://quran-com.p.rapidapi.com/chapter_recitations/{is_id}"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chapter_by_id(is_id: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a single chapter."
    id: Chapter ID ( 1-114)
        
    """
    url = f"https://quran-com.p.rapidapi.com/chapters/{is_id}"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def info(chapter_id: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get chapters information in specific language. Default to `English`."
    chapter_id: Chapter number ( 1-114 )
        
    """
    url = f"https://quran-com.p.rapidapi.com/chapters/{chapter_id}/info"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_chapters_by_reciters(chapter_number: int, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of chapters by reciters."
    id: Id of reciter for which you want to get the recitations. You can fetch list of all reciters from this endpoint

#endpoint:ZQvDmxKn7fQwLrAfy
        
    """
    url = f"https://quran-com.p.rapidapi.com/chapter_recitations/{is_id}/{chapter_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of Chapter (Surah) with the whole information of the number of verses, order, page etc."
    
    """
    url = f"https://quran-com.p.rapidapi.com/chapters"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verses_by_rub_el_hizb(rub_el_hizb_number: int, translations: str=None, words: str='true', language: str='en', audio: int=None, word_fields: str=None, tafsirs: str=None, fields: str=None, translation_fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all verses of a specific Rub el Hizb number(1-240)."
    rub_el_hizb_number: Rub el Hizb number(1-240)
        translations: comma separated ids of translations to load for each ayah.
        words: Include words of each ayah?

0 or false will not include words.

1 or true will include the words.
        language: Language to fetch word translation in specific language.
        audio: Id of recitation if you want to load audio of each ayah.
        word_fields: Comma separated list of word fields if you want to add more fields for each word. 
        tafsirs: Comma separated ids of tafisrs to load for each ayah if you want to load tafisrs.
        fields: comma separated  list of ayah fields.
        translation_fields: Comma separated list of translation fields if you want to add more fields for each translation. 
        
    """
    url = f"https://quran-com.p.rapidapi.com/verses/by_rub/{rub_el_hizb_number}"
    querystring = {}
    if translations:
        querystring['translations'] = translations
    if words:
        querystring['words'] = words
    if language:
        querystring['language'] = language
    if audio:
        querystring['audio'] = audio
    if word_fields:
        querystring['word_fields'] = word_fields
    if tafsirs:
        querystring['tafsirs'] = tafsirs
    if fields:
        querystring['fields'] = fields
    if translation_fields:
        querystring['translation_fields'] = translation_fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verses_by_juz(juz_number: int, translation_fields: str=None, language: str='en', audio: int=None, word_fields: str=None, translations: str=None, per_page: int=10, tafsirs: str=None, page: int=1, fields: str=None, words: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all verses from a specific juz(1-30)."
    translation_fields: Comma separated list of translation fields if you want to add more fields for each translation. 
        language: Language to fetch word translation in specific language.
        audio: Id of recitation if you want to load audio of each ayah.
        word_fields: Comma separated list of word fields if you want to add more fields for each word. 
        translations: comma separated ids of translations to load for each ayah.
        per_page: records per api call, you can get maximum 50 records. 
        tafsirs: Comma separated ids of tafisrs to load for each ayah if you want to load tafisrs.
        page: For paginating within the result
        fields: comma separated  list of ayah fields.
        words: Include words of each ayah?

0 or false will not include words.

1 or true will include the words.
        
    """
    url = f"https://quran-com.p.rapidapi.com/verses/by_juz/{juz_number}"
    querystring = {}
    if translation_fields:
        querystring['translation_fields'] = translation_fields
    if language:
        querystring['language'] = language
    if audio:
        querystring['audio'] = audio
    if word_fields:
        querystring['word_fields'] = word_fields
    if translations:
        querystring['translations'] = translations
    if per_page:
        querystring['per_page'] = per_page
    if tafsirs:
        querystring['tafsirs'] = tafsirs
    if page:
        querystring['page'] = page
    if fields:
        querystring['fields'] = fields
    if words:
        querystring['words'] = words
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(language: str='en', audio: int=None, translations: str=None, words: str='true', tafsirs: str=None, word_fields: str=None, translation_fields: str=None, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random verse. You can get random verse by `chapter`,`page`, `juz`, `hizb`, `rub el hizb`, `ruku`, `manzil` number, or from whole Quran."
    language: Language to fetch word translation in specific language.
        audio: Id of recitation if you want to load audio of each ayah.
        translations: comma separated ids of translations to load for each ayah.
        words: Include words of each ayah?

0 or false will not include words.

1 or true will include the words.
        tafsirs: Comma separated ids of tafisrs to load for each ayah if you want to load tafisrs.
        word_fields: Comma separated list of word fields if you want to add more fields for each word. 
        translation_fields: Comma separated list of translation fields if you want to add more fields for each translation. 
        fields: comma separated  list of ayah fields.
        
    """
    url = f"https://quran-com.p.rapidapi.com/verses/random"
    querystring = {}
    if language:
        querystring['language'] = language
    if audio:
        querystring['audio'] = audio
    if translations:
        querystring['translations'] = translations
    if words:
        querystring['words'] = words
    if tafsirs:
        querystring['tafsirs'] = tafsirs
    if word_fields:
        querystring['word_fields'] = word_fields
    if translation_fields:
        querystring['translation_fields'] = translation_fields
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def juz(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all juz"
    
    """
    url = f"https://quran-com.p.rapidapi.com/juzs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verses_by_chapter_id(chapter_number: int, tafsirs: str=None, word_fields: str=None, language: str='en', fields: str=None, words: str='true', per_page: int=10, page: int=1, translations: str=None, translation_fields: str=None, audio: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of verses by Chapter / Surah number."
    chapter_number: Chapter number ( 1  - 114 )
        tafsirs: Comma separated ids of tafisrs to load for each ayah if you want to load tafisrs.
        word_fields: Comma separated list of word fields if you want to add more fields for each word. 
        language: Language to fetch word translation in specific language.
        fields: comma separated  list of ayah fields.
        words: Include words of each ayah?

0 or false will not include words.

1 or true will include the words.
        per_page: records per api call, you can get maximum 50 records. 
        page: For paginating within the result
        translations: comma separated ids of translations to load for each ayah.
        translation_fields: Comma separated list of translation fields if you want to add more fields for each translation. 
        audio: Id of recitation if you want to load audio of each ayah.
        
    """
    url = f"https://quran-com.p.rapidapi.com/verses/by_chapter/{chapter_number}"
    querystring = {}
    if tafsirs:
        querystring['tafsirs'] = tafsirs
    if word_fields:
        querystring['word_fields'] = word_fields
    if language:
        querystring['language'] = language
    if fields:
        querystring['fields'] = fields
    if words:
        querystring['words'] = words
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if translations:
        querystring['translations'] = translations
    if translation_fields:
        querystring['translation_fields'] = translation_fields
    if audio:
        querystring['audio'] = audio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_hizb(recitation_id: int, hizb_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of ayah recitations for a Hizb."
    recitation_id: Recitation Id, you can get list of all ayah by ayah recitations using this endpoint #endpoint:HLbauN2sdGitPQPPL
        
    """
    url = f"https://quran-com.p.rapidapi.com/recitations/{recitation_id}/by_hizb/{hizb_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recitation_info_by_id(recitation_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information of a specific recitation. Could include detail about the reciter, recitation style. When and who recorded it etc."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/recitations/{recitation_id}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ayah_recitations(recitation_id: int, ayah_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of ayah recitations for a Hizb."
    recitation_id: Recitation Id, you can get list of all ayah by ayah recitations using this endpoint #endpoint:HLbauN2sdGitPQPPL
        ayah_key: Ayah key is combination of surah number and  ayah number. e.g 1:1 will be first Ayah of first Surah
        
    """
    url = f"https://quran-com.p.rapidapi.com/recitations/{recitation_id}/by_ayah/{ayah_key}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verse_media(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media related to the verse."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/verse_media"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tafsir_info_by_id(tafsir_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the information of a specific tafsir. Could include information about the author, when it was published etc."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/tafsirs/{tafsir_id}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all languages. You can get translated names of languages in specific language using `language` query parameter. For example
		
		  ```
		  /resources/languages?language=ur
		  ```
		
		will return language names translated into Urdu"
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/languages"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tafsirs(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available tafsirs."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/tafsirs"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translations(language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available translations. Use `language` query to get translated names of authors in specific language(e.g language=ur will send translation names in Urdu).
		
		For list of available language see  #endpoint:EZsWyDnekGaDKaCpt endpoint."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/translations"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recitations_by_chapter_number(chapter_number: int, recitation_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of ayah recitations for a Surah."
    recitation_id: Recitation Id, you can get list of all ayah by ayah recitations using this endpoint #endpoint:HLbauN2sdGitPQPPL
        
    """
    url = f"https://quran-com.p.rapidapi.com/recitations/{recitation_id}/by_chapter/{chapter_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chapters_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of chaper info we've in different languages."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/chapter_infos"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translation_info_by_id(translation_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information of a specific translation. Could include detail about the author, publishing date and publisher etc."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/translations/{translation_id}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recitations(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of available Recitations."
    language: Name of reciters in specific language. Will fallback to English if we don't have names in specific language.
        
    """
    url = f"https://quran-com.p.rapidapi.com/resources/recitations"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recitation_style(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the available recitation styles."
    
    """
    url = f"https://quran-com.p.rapidapi.com/resources/recitation_styles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recitations_by_juz_number(recitation_id: int, juz_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of ayah recitations for a juz."
    recitation_id: Recitation Id, you can get list of all ayah by ayah recitations using this endpoint #endpoint:HLbauN2sdGitPQPPL
        
    """
    url = f"https://quran-com.p.rapidapi.com/recitations/{recitation_id}/by_juz/{juz_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_text_without_tashkil_diacritical(chapter_number: int=None, hizb_number: int=None, verse_key: str=None, page_number: int=None, rub_el_hizb_number: int=None, juz_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Imlaei simple script(without tashkil/diacritical marks) of ayah."
    chapter_number: If you want to get text of a specific surah.
        hizb_number: If you want to get text of a specific hizb.
        verse_key: If you want to get text of a specific ayah.
        page_number: If you want to get text of a Madani Muhsaf page
        rub_el_hizb_number: If you want to get text of a specific Rub el Hizb.
        juz_number: If you want to get text of a specific juz.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/verses/imlaei"
    querystring = {}
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if verse_key:
        querystring['verse_key'] = verse_key
    if page_number:
        querystring['page_number'] = page_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if juz_number:
        querystring['juz_number'] = juz_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recitations_by_page_number(recitation_id: int, page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of ayah recitations for a Madani Mushaf page."
    recitation_id: Recitation Id, you can get list of all ayah by ayah recitations using this endpoint #endpoint:HLbauN2sdGitPQPPL
        
    """
    url = f"https://quran-com.p.rapidapi.com/recitations/{recitation_id}/by_page/{page_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def glyph_codes_qcf_v2_font(verse_key: str=None, hizb_number: int=None, rub_el_hizb_number: int=None, juz_number: int=None, chapter_number: int=None, page_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get glyph codes of ayah for QCF v2 font"
    verse_key: If you want to get text of a specific ayah.
        hizb_number: If you want to get text of a specific hizb.
        rub_el_hizb_number: If you want to get text of a specific Rub el Hizb.
        juz_number: If you want to get text of a specific juz.
        chapter_number: If you want to get text of a specific surah.
        page_number: If you want to get text of a Madani Muhsaf page
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/verses/code_v2"
    querystring = {}
    if verse_key:
        querystring['verse_key'] = verse_key
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if juz_number:
        querystring['juz_number'] = juz_number
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uthmani_text(verse_key: str=None, page_number: int=None, rub_el_hizb_number: int=None, hizb_number: int=None, chapter_number: int=None, juz_number: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Uthmani script of ayah. Use query strings to filter results, leave all query string blank if you want to fetch Uthmani script of whole Quran."
    verse_key: If you want to get Uthmani script of a specific ayah.
        page_number: If you want to get Uthmani script of a Madani Muhsaf page
        rub_el_hizb_number: If you want to get Uthmani script of a specific Rub el Hizb.
        hizb_number: If you want to get Uthmani script of a specific hizb.
        chapter_number: If you want to get Uthmani script of a specific surah.
        juz_number: If you want to get Uthmani script of a specific juz.
        
    """
    url = f"https://quran-com.p.rapidapi.com/quran/verses/uthmani"
    querystring = {}
    if verse_key:
        querystring['verse_key'] = verse_key
    if page_number:
        querystring['page_number'] = page_number
    if rub_el_hizb_number:
        querystring['rub_el_hizb_number'] = rub_el_hizb_number
    if hizb_number:
        querystring['hizb_number'] = hizb_number
    if chapter_number:
        querystring['chapter_number'] = chapter_number
    if juz_number:
        querystring['juz_number'] = juz_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quran-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

