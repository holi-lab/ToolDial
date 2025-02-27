import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_most_popular_authors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the most popular authors right now with a set of other relevant information."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/top_authors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_book_information_by_book_id(book_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Giving its Id, this endpoint returns a lot of information about that book, such as Name, Published Date, Authors, Synopsis, and many more."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/book/{book_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_weekly_popular_books_by_genre(genre: str, number_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most popular books from the last weeks given a genre."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/week/{genre}/{number_results}"
    querystring = {}
    if number_results:
        querystring['number_results'] = number_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_books_by_name(book_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Giving a book name (where spaces must be replaced by '+') you will receive a list of corresponding books. This list can return up to 20 books related to the input.
		Each book has information such as Name, Authors, Cover image URL, Launched Year, and many more."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/search/{book_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_awarded_books_of_a_year(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of the best books of a year by its winning category.
		Each returned book has its Name, Winning Category, Cover Image and more."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/top/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_nominated_books_for_a_genre_in_a_year(genre: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For a certain year and a genre, get the list of all nominated books to win an award."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/nominees/{genre}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_author_information_by_id(author_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a large variety of information of an Author such as the name, a brief biography, the list of books, and more."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/author/{author_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_top_15_most_popular_books_in_a_month_of_an_year(month: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide the year and the month (between 1 and 12) and receive the 15 most popular books for that month."
    
    """
    url = f"https://hapi-books.p.rapidapi.com/month/{year}/{month}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hapi-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

