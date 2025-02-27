import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_verses_starting_with_search_term(term1: str, first_book: str, not_sub_word_search_mode: bool=None, uppercase_mode: bool=None, term4: str=None, second_book: str='job', term2: str='with', term3: str=None, text_mode: str='full', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns verses starting with search term(s)"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/search_term/get_verses_starting_with_search_term"
    querystring = {'term1': term1, 'first_book': first_book, }
    if not_sub_word_search_mode:
        querystring['not_sub_word_search_mode'] = not_sub_word_search_mode
    if uppercase_mode:
        querystring['uppercase_mode'] = uppercase_mode
    if term4:
        querystring['term4'] = term4
    if second_book:
        querystring['second_book'] = second_book
    if term2:
        querystring['term2'] = term2
    if term3:
        querystring['term3'] = term3
    if text_mode:
        querystring['text_mode'] = text_mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_term_verse_address_summary(first_book: str, term1: str, word_search_mode: bool=None, term_filter_operator: str='and', term5: str=None, second_book: str='revelation', term2: str='fire', term3: str=None, term4: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a summary for verse addresses that contain term criteria.
		
		**  CAN SEARCH A RANGE OF BOOKS.  
		(first_book = 'matthew' , second_book = 'john' MEANS ENDPOINT SEARCHES  'matthew'  'mark'  'luke' 'john')    **"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/search_term/chapter_summary"
    querystring = {'first_book': first_book, 'term1': term1, }
    if word_search_mode:
        querystring['word_search_mode'] = word_search_mode
    if term_filter_operator:
        querystring['term_filter_operator'] = term_filter_operator
    if term5:
        querystring['term5'] = term5
    if second_book:
        querystring['second_book'] = second_book
    if term2:
        querystring['term2'] = term2
    if term3:
        querystring['term3'] = term3
    if term4:
        querystring['term4'] = term4
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_term_chapter_address_summary(first_book: str, term1: str, term_filter_operator: str='and', word_search_mode: bool=None, term4: str=None, term5: str=None, term2: str='fire', term3: str=None, second_book: str='revelation', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a summary for chapter addresses that contain term criteria.  A dictionary is returned using Bible book names as keys. Under each key, is a list of the chapters(in numerical form) that contains the term(s).
		
		**  CAN SEARCH A RANGE OF BOOKS.  
		(first_book = 'matthew' , second_book = 'john' MEANS ENDPOINT SEARCHES  'matthew'  'mark'  'luke' 'john')    **"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/search_term/chapter_summary"
    querystring = {'first_book': first_book, 'term1': term1, }
    if term_filter_operator:
        querystring['term_filter_operator'] = term_filter_operator
    if word_search_mode:
        querystring['word_search_mode'] = word_search_mode
    if term4:
        querystring['term4'] = term4
    if term5:
        querystring['term5'] = term5
    if term2:
        querystring['term2'] = term2
    if term3:
        querystring['term3'] = term3
    if second_book:
        querystring['second_book'] = second_book
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_term_count_occurrences_of_terms_found_in_text(term1: str, first_book: str, term4: str=None, term7: str=None, term5: str=None, term3: str=None, second_book: str='job', term2: str=None, term6: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**  CAN SEARCH A RANGE OF BOOKS.  
		(first_book = 'matthew' , second_book = 'john' MEANS ENDPOINT SEARCHES  'matthew'  'mark'  'luke' 'john')    **"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/search_term/counter"
    querystring = {'term1': term1, 'first_book': first_book, }
    if term4:
        querystring['term4'] = term4
    if term7:
        querystring['term7'] = term7
    if term5:
        querystring['term5'] = term5
    if term3:
        querystring['term3'] = term3
    if second_book:
        querystring['second_book'] = second_book
    if term2:
        querystring['term2'] = term2
    if term6:
        querystring['term6'] = term6
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_chapters_containing_terms(term1: str, first_book: str, uppercase_mode: bool=None, word_search_mode: bool=None, second_book: str=None, term_filter_operator: str='and', term2: str='light', term3: str=None, term4: str=None, text_mode: str='bionic', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes term(s) and returns chapters  that contain term(s). Terms are not case sensitive.  Books, operators etc. are NOT CASE SENSITIVE.
		
		**   ONLY SEARCHES 2 BOOKS   (NO RANGE FUNCTION) **"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/search_term/chapters"
    querystring = {'term1': term1, 'first_book': first_book, }
    if uppercase_mode:
        querystring['uppercase_mode'] = uppercase_mode
    if word_search_mode:
        querystring['word_search_mode'] = word_search_mode
    if second_book:
        querystring['second_book'] = second_book
    if term_filter_operator:
        querystring['term_filter_operator'] = term_filter_operator
    if term2:
        querystring['term2'] = term2
    if term3:
        querystring['term3'] = term3
    if term4:
        querystring['term4'] = term4
    if text_mode:
        querystring['text_mode'] = text_mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_verses_containing_terms(first_book: str, term1: str, uppercase_mode: bool=None, term_filter_operator: str='or', word_search_mode: bool=None, second_book: str='numbers', term2: str=None, term3: str=None, text_mode: str='full', term4: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes term(s) and returns verses that contain term(s). Terms are not case sensitive. 
		Books, operators etc. are NOT CASE SENSITIVE
		
		**  CAN SEARCH A RANGE OF BOOKS.  
		(first_book = 'matthew' , second_book = 'john' MEANS ENDPOINT SEARCHES  'matthew'  'mark'  'luke' 'john')    **"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/search_term/verses"
    querystring = {'first_book': first_book, 'term1': term1, }
    if uppercase_mode:
        querystring['uppercase_mode'] = uppercase_mode
    if term_filter_operator:
        querystring['term_filter_operator'] = term_filter_operator
    if word_search_mode:
        querystring['word_search_mode'] = word_search_mode
    if second_book:
        querystring['second_book'] = second_book
    if term2:
        querystring['term2'] = term2
    if term3:
        querystring['term3'] = term3
    if text_mode:
        querystring['text_mode'] = text_mode
    if term4:
        querystring['term4'] = term4
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chapter(book_name: str, uppercase_mode: bool=None, text_mode: str='vowels', chapter: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes parameters and returns one chapter for chosen bookname (ex: genesis, job, etc.  )"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/get_chapter"
    querystring = {'book_name': book_name, }
    if uppercase_mode:
        querystring['uppercase_mode'] = uppercase_mode
    if text_mode:
        querystring['text_mode'] = text_mode
    if chapter:
        querystring['chapter'] = chapter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_verses(text_mode: str, book_name: str, uppercase_mode: bool=None, verse_num2: int=5, verse_num1: int=1, chapter: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes parameters and returns requested verse OR Range of Verses in one of 8 text formats. 
		Example:   verse_num1 = 1 , verse_num2  = 5 will return verses( 1, 2, 3, 4, 5)"
    
    """
    url = f"https://bible-memory-verse-flashcard.p.rapidapi.com/get_verses"
    querystring = {'text_mode': text_mode, 'book_name': book_name, }
    if uppercase_mode:
        querystring['uppercase_mode'] = uppercase_mode
    if verse_num2:
        querystring['verse_num2'] = verse_num2
    if verse_num1:
        querystring['verse_num1'] = verse_num1
    if chapter:
        querystring['chapter'] = chapter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-memory-verse-flashcard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

