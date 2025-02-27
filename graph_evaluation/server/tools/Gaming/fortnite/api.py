import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def votinglistget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list with the items you've voted on. You can look back up to 7 days."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/voting/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usersidget(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/users/id"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creativegetget(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Interested in getting a specific island from our database? Use this endpoint."
    id: Island code of the island you want
        
    """
    url = f"https://fortnite1.p.rapidapi.com/creative/get"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcominggetget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive the unreleased items in this list. This list is updated every few minutes."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/upcoming/get"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemgetget(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you get access to data from a specific item. You can see, among other things, which set it belongs to and when the item was in the shop."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/item/get"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brmotdgetget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/br_motd/get"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemsrandomget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random items from our item database."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/items/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creativeaddget(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can add islands using this endpoint. The only thing you need is the code, we'll add the title and description automaticly.
		
		Images are possible to upload with a different endpoint, contact our team for the endpoint. All images will be manually approved by us, so abuse is not possible and any 18+ content won't pass our test.
		
		We're launching an edit API on Juli 1st, 2019."
    id: Island code to add
        
    """
    url = f"https://fortnite1.p.rapidapi.com/creative/add"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challengesgetget(season: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/challenges/get"
    querystring = {'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creativetagsget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/creative/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemspopularget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this API you get the popular items of the day, week and month. This is based on our ratings API."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/items/popular"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prod09userspublicbrstatsget(platform: str, user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "V1 of the stats endpoint. This endpoint can deliver stats for PC, PS4 and XBOX. Searching for Nintendo Switch, iOS and Android stats? Use V1."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/prod09/users/public/br_stats"
    querystring = {'platform': platform, 'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prod09userspublicmatchesv2get(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/prod09/users/public/matches_v2"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemslistget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a long list of all items in Fortnite, from outfits to wraps. Everything is listed."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/items/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def storegetget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the daily store that automatically updates at 1 am CET.
		
		**Example data for V2 (default version)**
		<pre><code>{
			"data": [
			    {
		            "itemId": "2fad344-834e456-dcf643d-91f9712",
		            "lastUpdate": 1553386039,
		            "store": {
		                "isFeatured": true,
		                "isRefundable": true,
		                "cost": "1500"
		            },
		            "item": {
		                "name": "Beastmode",
		                "description": "Gassed up and ready to roar.",
		                "type": "outfit",
		                "rarity": "epic",
		                "images": {
		                    "icon": "https://fortnite-public-files.theapinetwork.com/outfit/c567e52c290292c99c7c9b44dd36827c.png",
		                    "featured": "https://fortnite-public-files.theapinetwork.com/featured/2fad344-834e456-dcf643d-91f9712.png",
		                    "background": "https://fortnite-public-files.theapinetwork.com/image/2fad344-834e456-dcf643d-91f9712.png",
		                    "information": "https://fortnite-public-files.theapinetwork.com/image/2fad344-834e456-dcf643d-91f9712/icon.png"
		                },
		                "obtained_type": "vbucks",
		                "ratings": {
		                    "avgStars": 3.66,
		                    "totalPoints": 597,
		                    "numberVotes": 163
		                }
		            }
		        },
		        {
		        	/*/ SAME STRUCTURE AS ABOVE /*/
		        }
			]
		}</code></pre>"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/store/get"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creativelistget(order: str, tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "One big list (max 50 islands) with islands.
		
		You can sort using the following texts:
		popular
		views
		newest
		oldest
		rating
		
		Sorting on tags (for example prophunt or 1vs1) will be added Juli 1st, 2019."
    order: popular, views, newest, oldest or rating
        tag: (not required) only display islands with a specific tag (like deathrun, remakes or 1vs1)
        
    """
    url = f"https://fortnite1.p.rapidapi.com/creative/list"
    querystring = {'order': order, 'tag': tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def votingvoteget(itemid: str, stars: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vote on a specific item. An user can vote every 7 days on an item. 
		
		Example return:
		<pre><code>{
			"success": true,
		    "msg": "Successfully voted.",
		    "metadata": {
		        "votingId": "s3f318eec23af9e192ac9eabd0bc1c71",
		        "nextVote": 1554125012,
		        "ratings": {
		            "avgStars": 4.41,
		            "totalPoints": 9099,
		            "numberVotes": 2065
		        }
		    }
		}</code></pre>"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/voting/vote"
    querystring = {'itemid': itemid, 'stars': stars, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creativefetchget(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Thinking about adding Island to our system? That's possible, you can prefill your submit page using this endpoint.
		
		You'll also need the 'imageUploadId' from the response if you're willing to upload images."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/creative/fetch"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prod09userspublicbrstatsv2get(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "V2 of the stats endpoint. This endpoint includes stats for PC, PS4, XBOX, Android & iOS and Nintendo Switch stats.
		
		CONTROLLER = PS4 + Controller, XBOX + Controller, Switch + Controller or PC + Controller.
		
		TOUCHSCREEN = iOS, Android or Switch (WITHOUT CONTROLLER)
		
		KEYBOARD&MOUSE = PC with KB, PS4 with KB or XBOX with KB."
    
    """
    url = f"https://fortnite1.p.rapidapi.com/prod09/users/public/br_stats_v2"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weaponsgetget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/weapons/get"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stwmotdgetget(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    
    """
    url = f"https://fortnite1.p.rapidapi.com/stw_motd/get"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def creativesearchget(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TODO: Add Description"
    query: What are you searching?
        
    """
    url = f"https://fortnite1.p.rapidapi.com/creative/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnite1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

