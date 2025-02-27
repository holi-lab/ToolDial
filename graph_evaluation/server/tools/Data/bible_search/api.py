import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_book_by_name(bookname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a book of the Bible."
    bookname: Name of the book.
        
    """
    url = f"https://bible-search.p.rapidapi.com/books-by-name"
    querystring = {'bookName': bookname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chapter_by_bookname(bookname: str, chapterid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a chapter of a book in the Bible."
    bookname: Name of the book.
        chapterid: Chapter number.
        
    """
    url = f"https://bible-search.p.rapidapi.com/books-by-name/chapter"
    querystring = {'bookName': bookname, 'chapterId': chapterid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_verse_by_bookname(verseid: int, bookname: str, chapterid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a verse of the Bible."
    verseid: Verse number.
        bookname: Name of the book.
        chapterid: Chapter number.
        
    """
    url = f"https://bible-search.p.rapidapi.com/books-by-name/verse"
    querystring = {'verseId': verseid, 'bookName': bookname, 'chapterId': chapterid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_verses_by_bookname(chapterid: int, bookname: str, versestart: int, verseend: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a range of verses of the Bible."
    chapterid: Chapter number.
        bookname: Name of the book.
        versestart: Beginning verse.
        verseend: End verse.
        
    """
    url = f"https://bible-search.p.rapidapi.com/books-by-name/verses"
    querystring = {'chapterId': chapterid, 'bookName': bookname, 'verseStart': versestart, 'verseEnd': verseend, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_book_by_id(bookid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a book of the Bible."
    bookid: Id of the book.
        
    """
    url = f"https://bible-search.p.rapidapi.com/books-by-id"
    querystring = {'bookId': bookid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chapter_by_bookid(bookid: str, chapterid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a chapter of a book in the Bible."
    bookid: Id of the book.
        chapterid: Chapter number.
        
    """
    url = f"https://bible-search.p.rapidapi.com/books-by-id/chapter"
    querystring = {'bookId': bookid, 'chapterId': chapterid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_verse_by_bookid(chapterid: int, bookid: str, verseid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a verse of the Bible."
    chapterid: Chapter number.
        bookid: Id of the book.
        verseid: Verse number.
        
    """
    url = f"https://bible-search.p.rapidapi.com/books-by-id/verse"
    querystring = {'chapterId': chapterid, 'bookId': bookid, 'verseId': verseid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_text(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all verses from the bible by text parameter."
    text: Text to be searched.
        
    """
    url = f"https://bible-search.p.rapidapi.com/search"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_books(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of the books in the Bible."
    
    """
    url = f"https://bible-search.p.rapidapi.com/books"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_verse(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random verse of the bible."
    
    """
    url = f"https://bible-search.p.rapidapi.com/random-verse"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bible-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

