import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def motivational_quotes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The motivational quotes endpoint provides users with inspiring quotes from notable figures that can help motivate and encourage them to achieve their goals. Each quote is accompanied by the name of the author or speaker, giving users the opportunity to learn from and be inspired by some of the greatest minds in history. These quotes are designed to uplift and empower users, reminding them that they have the strength and ability to overcome any obstacle and accomplish their dreams."
    
    """
    url = f"https://quotable-api-motivational-funny-and-movie-quotes.p.rapidapi.com/motivational_quotes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotable-api-motivational-funny-and-movie-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_quotes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The movie quotes endpoint of the API returns a random quote from a popular movie along with the name of the movie. The response is in JSON format, with the quote and the movie name as separate key-value pairs. The endpoint is designed to provide users with a way to access and display entertaining and memorable quotes from their favorite movies."
    
    """
    url = f"https://quotable-api-motivational-funny-and-movie-quotes.p.rapidapi.com/movie_quotes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotable-api-motivational-funny-and-movie-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def funny_quotes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The funny quotes endpoint is a part of an API that provides a collection of funny quotes. This endpoint specifically allows users to access a random funny quote from the collection or retrieve all of the available funny quotes. Users can send a GET request to the endpoint to receive a response in either JSON, which includes the text of the quote and any relevant metadata, such as the author or the category of the quote. This endpoint is designed to provide users with a source of humor and entertainment, and can be integrated into various applications, websites, or services to add a lighthearted touch to user experiences."
    
    """
    url = f"https://quotable-api-motivational-funny-and-movie-quotes.p.rapidapi.com/funny_quotes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotable-api-motivational-funny-and-movie-quotes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

