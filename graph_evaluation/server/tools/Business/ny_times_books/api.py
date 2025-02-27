import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_lists_overview_format(published_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top 5 books for all the Best Sellers lists for specified date."
    published_date: YYYY-MM-DD

The best-seller list publication date.
You do not have to specify the exact date the list was published. The service will search forward (into the future) for the closest publication date to the date you specify. For example, a request for lists/overview/2013-05-22 will retrieve the list that was published on 05-26.

If you do not include a published date, the current week's best sellers lists will be returned.
        
    """
    url = f"https://ny-times-books.p.rapidapi.com/lists/overview.json"
    querystring = {}
    if published_date:
        querystring['published_date'] = published_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lists_best_sellers_history_json(publisher: str=None, price: str=None, offset: int=None, author: str=None, title: str=None, age_group: str=None, contributor: str=None, isbn: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Best Sellers list history."
    publisher: The standardized name of the publisher
        price: The publisher's list price of the best seller, including decimal point.
        offset: Sets the starting point of the result set (0, 20, ...).  Used to paginate thru results if there are more than 20. Defaults to 0. The num_results field indicates how many results there are total.
        author: The author of the best seller. The author field does not include additional contributors (see Data Structure for more details about the author and contributor fields).

When searching the author field, you can specify any combination of first, middle and last names.

When sort-by is set to author, the results will be sorted by author's first name.
        title: The title of the best seller

When searching, you can specify a portion of a title or a full title.
        age_group: The target age group for the best seller.
        contributor: The author of the best seller, as well as other contributors such as the illustrator (to search or sort by author name only, use author instead).

When searching, you can specify any combination of first, middle and last names of any of the contributors.

When sort-by is set to contributor, the results will be sorted by the first name of the first contributor listed.
        isbn: International Standard Book Number, 10 or 13 digits

A best seller may have both 10-digit and 13-digit ISBNs, and may have multiple ISBNs of each type. To search on multiple ISBNs, separate the ISBNs with semicolons (example: 9780446579933;0061374229).
        
    """
    url = f"https://ny-times-books.p.rapidapi.com/lists/best-sellers/history.json"
    querystring = {}
    if publisher:
        querystring['publisher'] = publisher
    if price:
        querystring['price'] = price
    if offset:
        querystring['offset'] = offset
    if author:
        querystring['author'] = author
    if title:
        querystring['title'] = title
    if age_group:
        querystring['age-group'] = age_group
    if contributor:
        querystring['contributor'] = contributor
    if isbn:
        querystring['isbn'] = isbn
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lists_full_overview_format(published_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all books for all the Best Sellers lists for specified date."
    published_date: YYYY-MM-DD

The best-seller list publication date.
You do not have to specify the exact date the list was published. The service will search forward (into the future) for the closest publication date to the date you specify. For example, a request for lists/overview/2013-05-22 will retrieve the list that was published on 05-26.

If you do not include a published date, the current week's best sellers lists will be returned.
        
    """
    url = f"https://ny-times-books.p.rapidapi.com/lists/full-overview.json"
    querystring = {}
    if published_date:
        querystring['published_date'] = published_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lists_names_format(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Best Sellers list names."
    
    """
    url = f"https://ny-times-books.p.rapidapi.com/lists/names.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lists_format(list: str, offset: int=None, bestsellers_date: str=None, published_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Best Sellers list.  If no date is provided returns the latest list."
    list: The name of the Times best sellers list (hardcover-fiction, paperback-nonfiction, ...).
The /lists/names service returns all the list names.
The encoded list names are lower case with hyphens instead of spaces (e.g. e-book-fiction, instead of E-Book Fiction).
        offset: Sets the starting point of the result set (0, 20, ...).  Used to paginate thru books if list has more than 20. Defaults to 0.  The num_results field indicates how many books are in the list.
        bestsellers_date: YYYY-MM-DD

The week-ending date for the sales reflected on list-name. Times best sellers lists are compiled using available book sale data. The bestsellers-date may be significantly earlier than published-date. For additional information, see the explanation at the bottom of any best-seller list page on NYTimes.com (example: Hardcover Fiction, published Dec. 5 but reflecting sales to Nov. 29).
        published_date: YYYY-MM-DD

The date the best sellers list was published on NYTimes.com (different than bestsellers-date).  Use "current" for latest list.
        
    """
    url = f"https://ny-times-books.p.rapidapi.com/lists.json"
    querystring = {'list': list, }
    if offset:
        querystring['offset'] = offset
    if bestsellers_date:
        querystring['bestsellers-date'] = bestsellers_date
    if published_date:
        querystring['published-date'] = published_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lists_date_list_json(list: str, date: str, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Best Sellers list by date."
    list: Name of the Best Sellers List (e.g. hardcover-fiction). You can get the full list of names from the /lists/names.json service.
        date: YYYY-MM-DD or "current"

The date the best sellers list was published on NYTimes.com.  Use "current" to get latest list.
        offset: Sets the starting point of the result set (0, 20, ...).  Used to paginate thru books if list has more than 20. Defaults to 0.  The num_results field indicates how many books are in the list.
        
    """
    url = f"https://ny-times-books.p.rapidapi.com/lists/{date}/{list}.json"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reviews_format(author: str=None, isbn: int=None, title: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get book reviews."
    author: You’ll need to enter the author’s first and last name, separated by a space. This space will be converted into the characters %20.
        isbn: Searching by ISBN is the recommended method. You can enter 10- or 13-digit ISBNs.
        title: You’ll need to enter the full title of the book. Spaces in the title will be converted into the characters %20.
        
    """
    url = f"https://ny-times-books.p.rapidapi.com/reviews.json"
    querystring = {}
    if author:
        querystring['author'] = author
    if isbn:
        querystring['isbn'] = isbn
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-books.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

