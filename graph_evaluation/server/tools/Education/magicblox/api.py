import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def book(accept_encoding: str='gzip', seed: int=853486, reading_level: str=None, offset: int=0, nid: int=None, limit: int=20, author: str=None, section: str=None, language: str=None, title: str=None, category: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Two ways to use this endpoint:
		1. Search by category, language, reading level, section, title, author, or any combination of these filters. This will give you a list of books to discover, along with metadata, a cover image for each book, and the book's NID number.
		
		2. Use the NID of a book (with no other parameters) to get the content pages of the book, along with its metadata and cover image. This will only work for a single NID at a time. This is counted as a "read" and counts toward your quota. **Important: You must send us your watermark in order to access this content. More information below.**
		
		## Required: Your watermark
		Before you can see the contents of our books, you must send us an image we can use to watermark each page (usually your company's logo). We do this to protect our content creators. Please send your watermark in .png format to apisupport@magicblox.com. Images should be no more than 100 pixels high and no more than 200 pixels wide and should have a transparent background."
    accept_encoding: Requests GZIP compression from the CDN by providing 'gzip' as a value. This is strongly recommended for queries and when loading images to decrease loading times and lower bandwidth of book content.
        seed: Specify a seed (between 1 and1000000) to randomize your results. Results randomized in this way will always return in the same order, so you can paginate (see limit and offset) a randomly ordered set of results without seeing the same book twice or missing a book.
        reading_level: Returns a specified books section.

**Possible values:**
- 'Beginner (1 to 6)'
- 'Intermediate (6 to 9)'
- 'Advanced (9 to 13)'
        offset: Sets to offset of the books returned. Use this in collaboration with the 'limit' parameter to implement pagination.

For example, if your search results in 100 books, you could retrieve all of them by specifying a limit of 20 and making five queries with offsets 0, 20, 40, 60, and 80.
        nid: Use the NID of a book (with no other parameters) to get the content pages of the book, along with its metadata and cover image. This will only work for a single NID at a time. This is counted as a \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"read\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" and counts toward your quota. **Important: You must send us your watermark in order to access this content. More information below.**

### Required: Your watermark
But before you can see the contents of our books, you must send us an image we can use to watermark each page (usually your company's logo). We do this to protect our content creators. Please send your watermark in .png format to apisupport@magicblox.com. Images should be no more than 100 pixels high and no more than 200 pixels wide and should have a transparent background.
        limit: Limits the number of books returned by the query. Use this in collaboration with the 'offset' parameter to implement pagination.

For example, if your search results in 100 books, you could retrieve all of them by specifying a limit of 20 and making five queries with offsets 0, 20, 40, 60, and 80.
        author: Returns books with authors similar to author param.
        section: Returns a specified books section.

**Possible values:**
- 'all'
- 'award-winning'
- 'featured'
- 'most-read'
- 'recently-added'
        language: Restricts results to the specified language.

**Possible values:**
- 'English'
- 'Spanish'
- 'Italian'
- 'French'
- 'German'
- 'Mongolian'
- 'Portuguese'
- 'Hebrew'
        title: Returns books with titles similar to the title param.
        category: Specify a category by tid obtained from the /categories endpoint.
        
    """
    url = f"https://magicblox.p.rapidapi.com/v1/book"
    querystring = {}
    if accept_encoding:
        querystring['Accept-Encoding'] = accept_encoding
    if seed:
        querystring['seed'] = seed
    if reading_level:
        querystring['reading_level'] = reading_level
    if offset:
        querystring['offset'] = offset
    if nid:
        querystring['nid'] = nid
    if limit:
        querystring['limit'] = limit
    if author:
        querystring['author'] = author
    if section:
        querystring['section'] = section
    if language:
        querystring['language'] = language
    if title:
        querystring['title'] = title
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magicblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sandbox(title: str=None, author: str=None, nid: int=None, reading_level: str=None, category: str=None, language: str=None, section: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint intended for testing the API. This works exactly like the /book endpoint but without the quota limitations.
		
		## Two ways to use this endpoint:
		1. Search by category, language, reading level, section, title, author, or any combination of these filters. This will give you a list of books to discover, along with metadata, a cover image for each book, and the book's NID number.
		
		2. Use the NID of a book (with no other parameters) to get the content pages of the book, along with its metadata and cover image. This will only work for a single NID at a time."
    title: Returns books with titles similar to the title param.
        author: Returns books with authors similar to author param.
        nid: Use the NID of a book (with no other parameters) to get the content pages of the book, along with its metadata and cover image. This will only work for a single NID at a time.
        reading_level: Returns a specified books section.

**Possible values:**
- 'Beginner (1 to 6)'
- 'Intermediate (6 to 9)'
- 'Advanced (9 to 13)'
        category: Specify a category by tid obtained from the /categories endpoint.
        language: Restricts results to the specified language.

**Possible values:**
- 'English'
- 'Spanish'
- 'Italian'
- 'French'
- 'German'
- 'Mongolian'
- 'Portuguese'
        section: Returns a specified books section.

**Possible values:**
- 'all'
- 'award-winning'
- 'featured'
- 'most-read'
- 'recently-added'
        
    """
    url = f"https://magicblox.p.rapidapi.com/v1/sandbox"
    querystring = {}
    if title:
        querystring['title'] = title
    if author:
        querystring['author'] = author
    if nid:
        querystring['nid'] = nid
    if reading_level:
        querystring['reading_level'] = reading_level
    if category:
        querystring['category'] = category
    if language:
        querystring['language'] = language
    if section:
        querystring['section'] = section
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magicblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of possible categories to be used in the book and sandbox endpoints"
    
    """
    url = f"https://magicblox.p.rapidapi.com/v1/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magicblox.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

