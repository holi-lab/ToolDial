import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_author_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data about the author.
		Slower than /authors/author-id/{author-id}."
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/authors/user-id/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_author_by_author_id(author_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data about the author.
		Faster than /authors/user-id/{user-id}."
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/authors/author-id/{author_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_author_s_books_by_author_id(author_id: int, sort: str='publication-year', page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all books written by the author with the provided author ID.
		Each page contains 30 results."
    sort: Sort by:
- title
- rating
- popularity
- publication-year
- number-of-pages
        
    """
    url = f"https://goodreads-books.p.rapidapi.com/authors/{author_id}/books"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_authors(q: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all the authors that match the query.
		Each page contains 20 results."
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/authors/search"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_books(isbn: int=None, q: str='harry', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all books that matches the query (title, author, ISBN, tag,  genre...).
		Each page contains 20 results."
    isbn: If it is provided, 'q' and 'page' query parameters will have no effect.
Returns the book that matches that ISBN.
        
    """
    url = f"https://goodreads-books.p.rapidapi.com/search"
    querystring = {}
    if isbn:
        querystring['isbn'] = isbn
    if q:
        querystring['q'] = q
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all books included on a list (100 books per page)
		You can get the id of a list using the endpoint "/lists" to retrieve all available lists, or visiting [goodreads.com/list](https://www.goodreads.com/list) to search for a list and copying it's id from the url."
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/lists/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lists(page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the best lists of books (each page contains 10 lists)"
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/lists"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def best_books(genre: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of the 50 most representative books of a genre
		To see the available genres go to [goodreads.com/genres/list](https://www.goodreads.com/genres/list?utf8=%E2%9C%93&filter=top-level)."
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/genres/{genre}/best"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def book_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data about a particular book (author, reviews, genres...)"
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/books/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def newest_books(genre: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of the 100 newest releases of a particular genre.
		To see the available genres go to [goodreads.com/genres/list](https://www.goodreads.com/genres/list?utf8=%E2%9C%93&filter=top-level).
		Note: Some genres may not have new releases."
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/genres/{genre}/newest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def books_of_the_week(genre: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of the 100 books most read this week of a genre
		To see the available genres go to [goodreads.com/genres/list](https://www.goodreads.com/genres/list?utf8=%E2%9C%93&filter=top-level).
		Note: Some genres may not have a "most read this week" list."
    
    """
    url = f"https://goodreads-books.p.rapidapi.com/genres/{genre}/week"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "goodreads-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

