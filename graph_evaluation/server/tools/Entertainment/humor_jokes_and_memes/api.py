import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_joke(topics: str, max_length: int=1000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a joke using a large language model."
    
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/jokes/create"
    querystring = {'topics': topics, }
    if max_length:
        querystring['max-length'] = max_length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insult(reason: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insult somebody for doing something.
		See https://humorapi.com/docs/#Insult for more."
    reason: The reason for the praise/insult.
        name: The person's name.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/insult"
    querystring = {'reason': reason, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def praise(reason: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Praise somebody for doing something.
		See https://humorapi.com/docs/#Praise for more."
    reason: The reason for the praise/insult.
        name: The person's name.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/praise"
    querystring = {'reason': reason, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rate_word(word: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate the funniness of a word.
		See https://humorapi.com/docs/#Rate-Word for more."
    word: The word to be rated.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/words/rate"
    querystring = {'word': word, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_nonsense_word(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a nonsense word.
		See https://humorapi.com/docs/#Generate-Nonsense-Word for more."
    
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/words/nonsense/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_gifs(query: str, number: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for gifs.
		See https://humorapi.com/docs/#Search-Gifs for more."
    query: A search query.
        number: The number of results to retrieve between 1 and 10.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/gif/search"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_meme(keywords: str='rocket', number: int=3, media_type: str='image', keywords_in_image: bool=None, min_rating: int=4, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random meme.
		See https://humorapi.com/docs/#Random-Meme for more."
    keywords: A comma separated list of keywords.
        number: The number of results to retrieve between 1 and 10.
        media_type: The type of the content. Can be either 'image' or 'video' or specific formats such as 'jpg', 'png', 'gif', or 'mp4'.
        keywords_in_image: Whether the keywords should be found in the meme's image.
        min_rating: The minimum rating between 0 and 10 the result should have.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/memes/random"
    querystring = {}
    if keywords:
        querystring['keywords'] = keywords
    if number:
        querystring['number'] = number
    if media_type:
        querystring['media-type'] = media_type
    if keywords_in_image:
        querystring['keywords-in-image'] = keywords_in_image
    if min_rating:
        querystring['min-rating'] = min_rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_memes(media_type: str='image', keywords_in_image: bool=None, keywords: str='rocket', min_rating: int=3, number: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for memes.
		See https://humorapi.com/docs/#Search-Memes for more."
    media_type: The type of the content. Can be either 'image' or 'video' or specific formats such as 'jpg', 'png', 'gif', or 'mp4'.
        keywords_in_image: Whether the keywords should be found in the meme's image.
        keywords: A comma separated list of keywords.
        min_rating: The minimum rating between 0 and 10 the result should have.
        number: The number of results to retrieve between 1 and 10.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/memes/search"
    querystring = {}
    if media_type:
        querystring['media-type'] = media_type
    if keywords_in_image:
        querystring['keywords-in-image'] = keywords_in_image
    if keywords:
        querystring['keywords'] = keywords
    if min_rating:
        querystring['min-rating'] = min_rating
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jokes(number: int=3, exclude_tags: str='nsfw', max_length: int=200, include_tags: str='one_liner', keywords: str='rocket', min_rating: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for jokes.
		See https://humorapi.com/docs/#Search-Jokes for more."
    number: The number of results to retrieve between 1 and 10.
        exclude_tags: A comma separated list of tags that the joke must not have.
        max_length: The maximum number of letters in the joke.
        include_tags: A comma separated list of tags that the joke must have.
        keywords: A comma separated list of keywords.
        min_rating: The minimum rating between 0 and 10 the result should have.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/jokes/search"
    querystring = {}
    if number:
        querystring['number'] = number
    if exclude_tags:
        querystring['exclude-tags'] = exclude_tags
    if max_length:
        querystring['max-length'] = max_length
    if include_tags:
        querystring['include-tags'] = include_tags
    if keywords:
        querystring['keywords'] = keywords
    if min_rating:
        querystring['min-rating'] = min_rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_joke(keywords: str='rocket', include_tags: str='one_liner', min_rating: int=7, max_length: int=200, exclude_tags: str='nsfw', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random joke.
		See https://humorapi.com/docs/#Random-Joke for more."
    keywords: A comma separated list of keywords.
        include_tags: A comma separated list of tags that the joke must have.
        min_rating: The minimum rating between 0 and 10 the result should have.
        max_length: The maximum number of letters in the joke.
        exclude_tags: A comma separated list of tags that the joke must not have.
        
    """
    url = f"https://humor-jokes-and-memes.p.rapidapi.com/jokes/random"
    querystring = {}
    if keywords:
        querystring['keywords'] = keywords
    if include_tags:
        querystring['include-tags'] = include_tags
    if min_rating:
        querystring['min-rating'] = min_rating
    if max_length:
        querystring['max-length'] = max_length
    if exclude_tags:
        querystring['exclude-tags'] = exclude_tags
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "humor-jokes-and-memes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

