import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_books(book_type: str='Fiction', categories: str=None, lexile_max: int=800, series: str='Wings of fire', lexile_min: int=600, page: int=1, author: str=None, title: str=None, results_per_page: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search books using optional title, author, series, book type, categories, and [lexile measure](https://lexile.com/educators/understanding-lexile-measures/) range."
    book_type: Book type, fiction or nonfiction.
        categories: Semicolon separated book categories to search for, e.g., 'Mystery & Suspense; Science Fiction & Fantasy'. Available categories include: ['Animals, Bugs & Pets', 'Art, Creativity & Music', 'General Literature', 'Hobbies, Sports & Outdoors', 'Science Fiction & Fantasy', 'Real Life', 'Science & Technology', 'Mystery & Suspense', 'Reference']
        lexile_max: Maximum Lexile measure.
        series: Series name.
        lexile_min: Minimum Lexile measure.
        page: Page of the results.
        author: Name of the author.
        title: Terms to be appeared in book title.
        results_per_page: Results per page.
        
    """
    url = f"https://book-finder1.p.rapidapi.com/api/search"
    querystring = {}
    if book_type:
        querystring['book_type'] = book_type
    if categories:
        querystring['categories'] = categories
    if lexile_max:
        querystring['lexile_max'] = lexile_max
    if series:
        querystring['series'] = series
    if lexile_min:
        querystring['lexile_min'] = lexile_min
    if page:
        querystring['page'] = page
    if author:
        querystring['author'] = author
    if title:
        querystring['title'] = title
    if results_per_page:
        querystring['results_per_page'] = results_per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-finder1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

