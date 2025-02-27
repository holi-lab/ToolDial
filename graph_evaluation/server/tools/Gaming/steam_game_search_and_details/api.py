import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def game_details_from_steam_game_id_or_app_id(search_value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass in the steam game id and retrieve the details for the game. 
		
		**Returned Items:**
		- title : The Game Title
		- category : The first category found in popular_tags
		- md5 : MD5 from the title
		- game_description_snippet : Short text used to describe the game.
		- game_review_summary : Text showing if the overall reviews are positive or negative
		- responsive_reviewdesc : Text showing the percentage of reivews and the number with positive or negative. 
		- review_number : The number of total reviews.
		- release_date : The date the game was released
		- developer : Game Developer
		- publisher : Game Publisher
		- popular_tags : all of the tags used to categorise the game
		- video_webm : the 1st video url found for this game
		- video_webm_hd : the 1st video url found for this game in HD (where available)
		- video_poster : the poster image url used for the 1st video
		- image_highlight : the 1st image url found in large res
		- game_area_description : the full HTML for the game description area
		- sys_req : if system requirements are found for the game
		- community_hub_url : url to the games community discussions hub
		- game_url : the games url
		- game_id : the game id"
    
    """
    url = f"https://steam-game-search-and-details.p.rapidapi.com/game_details/search_like/game_id/"
    querystring = {'search_value': search_value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-game-search-and-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_details_from_steam_url(search_value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can pass in a Steam app or game url here and it will return the details. 
		
		**Returned Items:**
		- title : The Game Title
		- category : The first category found in popular_tags
		- md5 : MD5 from the title
		- game_description_snippet : Short text used to describe the game.
		- game_review_summary : Text showing if the overall reviews are positive or negative
		- responsive_reviewdesc : Text showing the percentage of reivews and the number with positive or negative. 
		- review_number : The number of total reviews.
		- release_date : The date the game was released
		- developer : Game Developer
		- publisher : Game Publisher
		- popular_tags : all of the tags used to categorise the game
		- video_webm : the 1st video url found for this game
		- video_webm_hd : the 1st video url found for this game in HD (where available)
		- video_poster : the poster image url used for the 1st video
		- image_highlight : the 1st image url found in large res
		- game_area_description : the full HTML for the game description area
		- sys_req : if system requirements are found for the game
		- community_hub_url : url to the games community discussions hub
		- game_url : the games url
		- game_id : the game id"
    
    """
    url = f"https://steam-game-search-and-details.p.rapidapi.com/game_details/search_like/game_url/"
    querystring = {'search_value': search_value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-game-search-and-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_search_by_game_title(search_value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This allows you to search for a game by its title, the returned result matches games similar to the title and also matches the title, but the returned list will still need to be checked for the correct game title. 
		
		**Returned Items:**
		
		- title: Title of the game
		- category (search term)
		- md5 (from title)
		- image_thumbnail
		- search_released : Game Release Date
		- search_review_summary : if there is comments on the reviews
		- search_price : Price at the time of the original search in USD
		- search_discount : If there is any discount on the item at the time of original search.
		- details_url : the details link (can be passed into the Game details from Steam URL Endpoint)
		- search_term : the search term that was used to find this item
		- game_id : the id of the game (can be passed into the Game Details from Steam game id Endpoint)"
    
    """
    url = f"https://steam-game-search-and-details.p.rapidapi.com/game_search/search_like/title/"
    querystring = {'search_value': search_value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-game-search-and-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

