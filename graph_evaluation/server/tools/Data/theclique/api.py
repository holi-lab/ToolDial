import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def talent_list(page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The data is divided into 26 pages of 500 records. To get data, you need to specify the required page"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/the_right_fit/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventbrite_california(start_date: str, page_num: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Events for next weekend in California. There is a pagination of 50 events."
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/event_brite_california/{start_date}/{page_num}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_short_info(artists_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get brief information on several artists at the same time. It is enough to enter the ID of the artists through a comma"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/spotify_short_info"
    querystring = {'artists_id': artists_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genres(country: str, genre_id: str, content_limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For each country, the content of genres is different. You can get genres for almost any country. You just need to enter in the field 'country' Alpha-2 code. For example, Australia - AU, United Kingdom - UK, USA - US"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/spotify_genre/{genre_id}/{content_limit}/{limit}/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agency_clients(slug: str='wasserman', agency_id: int=440, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an opportunity to obtain data on players and managers (if any) whose representative is the required agency. There is a pagination. 25 clients on one page. Pagination is applied immediately to both players and managers. 
		For example, the agency has 5 pages with players and 2 pages with managers. If we choose page = 2, we get data from the 2nd page of players and the 2nd page of managers. If we choose page = 4, we will get data from the 4th page of players and the 2nd page of managers"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarkt/agency_players"
    querystring = {}
    if slug:
        querystring['slug'] = slug
    if agency_id:
        querystring['agency_id'] = agency_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def movie_like_more(movie_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter movie id"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb_movies/{movie_id}"
    querystring = {}
    if movie_id:
        querystring['movie_id'] = movie_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def imdb_company_clients(id_company: str, page_int: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get the company's clients. If the company has more than 500 clients, then you need to use pagination.
		page_int allows you to get the next 500 clientsof the company. For example, 0 is the first 500 clients, 1 is the next 500, and so on"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb/company_clients/{id_company}/{page_int}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chemistware_house(product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "chemistware_house product"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/chemistware_house/{product_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def igashop(product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Igashop product"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/igashop/{product_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def woolworths(product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Woolworths product"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/woolworths/{product_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def imdb_pro_company(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View the contacts of a specific company"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb/company/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def imdb_pro_person(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View the contacts of a specific artist or other employee"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb/name/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shazam_album(album_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the album ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/shazam/album/{album_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shazam_artist(artist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the Artist ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/shazam/artist/{artist_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shazam_search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the name of the artist"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/shazam/search/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos(id_human: str, sort_by: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter artist id"
    sort_by: sorting options:
- featured
- most_recent
- longest
- shortest
- expiring
        
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb/videos/{id_human}/{sort_by}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filmography(id_human: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter artist id"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb/filmography/{id_human}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_cast(id_title: str, filter_num: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter Title ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb/titles/top_cast/{id_title}/{filter_num}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(id_human: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter talent id"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/imdb/images/{id_human}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fanspro_overview(is_id: str, type: str, season: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter Type (agent, player, team), id
		When choosing a team, you can choose a season (2022, 2021, ...)"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/fanspro"
    querystring = {'id': is_id, 'type': type, }
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playlist(channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlist by Channel ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/youtube/playlist"
    querystring = {'channel_id': channel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(geo: str, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending videos of the particular geo"
    geo: ISO 3166-2 country code of the region for which you want the trending data. Like US (default), UK, CA, IN, etc.
        
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/youtube/trending"
    querystring = {'geo': geo, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel(channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel info by ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/youtube/channel"
    querystring = {'channel_id': channel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def oztix(event_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You just need to enter the name of the event or artist. You will receive information about all available events
		
		For example:
		BEST NIGHT EVER - Hockey Dad & Ruby Fields - Gold Coast
		or
		Adam Newling - 'Barmy' Tour
		or
		Good Things Festival 2022"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/oztix/{event_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moshtix(event_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You just need to enter the name of the event or artist. You will receive information about all available events.
		A few clarifications to get the correct answer:
		
		1. If the name contains the symbol + or /, please shorten the name to the beginning of this symbol. For example, searching by name "Y&O Sundays w/ Vetta Borne // Ra Ra Viper // Edith // Ricky's Breath" will not give results. But if will use shorted name "Y&O Sundays", you will get the desired result.
		
		2. If the name is too long, try to shorten it sensibly. For example, searching by name "SEADECK SYDNEY - SOUNDS LIKE SUNDAYS FT. J. WORRA - Sunday 20th November" will not give results. But if will use shorted name "SEADECK SYDNEY - SOUNDS LIKE SUNDAYS", you will get the desired result.
		
		The reasons for these clarifications arise from the specific work of the Moshtix site search engine. 
		Thank you for your understanding and I wish you pleasant work ;)"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/moshtix/{event_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_nation(event_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For get results input ***slug***
		For example, from url "https://www.livenation.com.au/festival/united-we-dance-tickets" need get only ***united-we-dance-tickets***
		Works only for festivals"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/live_nation/{event_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resident_advisor_api(event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter ID event
		
		For example, with the url "https://ra.co/events/1618892" we need "1618892""
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/resident_advisor/{event_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_by_club(slug: str, season: int, club_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG, ID and SEASON
		
		For example: chelsea-fc, 631, 2022"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarkt/games"
    querystring = {'slug': slug, 'season': season, 'club_id': club_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_achievements(slug: str, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG and PLAYER_ID
		
		For example: lionel-messi, 28003"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarkt/achievement"
    querystring = {'slug': slug, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_top_artists(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Top Artists chart.
		Available years **2006 - 2022**"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/yearly/top-artists"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_billboard_global_200(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Billboard Global 200 chart.
		Available years **2021 - 2022**"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/yearly/billboard-global-200"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_billboard_200_albums(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Billboard 200 Albums chart.
		Available years **2002 - 2022**"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/yearly/top-billboard-200-albums"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def year_end_hot_100_songs(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Year-End Hot 100 Songs chart.
		Available years **2006 - 2022**"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/yearly/hot-100-songs"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_artist_100(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Artist 100 chart."
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/weekly/artist-100"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_global_200(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Billboard Global 200 chart."
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/weekly/billboard-global-200"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_billboard_200(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Billboard 200 chart."
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/weekly/billboard-200"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_hot_100(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Billboard Hot 100 chart."
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/billboard/weekly/hot-100"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG of the artist
		You can get SLUG from Search endpoint"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/last_fm/events"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def overview(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detailed info about Artist by SLUG
		You can get SLUG from Search endpoint"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/last_fm/overview"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint for search artists by NAME"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/last_fm/search"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agency(slug: str='spocs-global-sports', agency_id: int=728, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG and ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarkt/agency"
    querystring = {}
    if slug:
        querystring['slug'] = slug
    if agency_id:
        querystring['agency_id'] = agency_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def everything(domains: str=None, q: str='the weeknd', exclude_domains: str=None, language: str='en', from_param: str=None, sort_by: str=None, to: str=None, page: str=None, page_size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through millions of articles from over 80,000 large and small news sources and blogs.
		
		This endpoint suits article discovery and analysis."
    domains: A comma-seperated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to restrict the search to. 
        q: 

Keywords or phrases to search for in the article title and body.

Advanced search is supported here:

    Surround phrases with quotes (\\\\\\\") for exact match.
    Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
    Prepend words that must not appear with a - symbol. Eg: -bitcoin
    Alternatively you can use the AND / OR / NOT keywords, and optionally group these with parenthesis. Eg: crypto AND (ethereum OR litecoin) NOT bitcoin.

The complete value for q must be URL-encoded. Max length: 500 chars.
        exclude_domains: A comma-seperated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to remove from the results. 
        language: The 2-letter ISO-639-1 code of the language you want to get headlines for. Possible options: **ar**, **de**, **en**, **es**, **fr**, **he**, **it**, **nl**, **no**, **pt**, ~~ru~~, **sv**, **ud**, **zh**.

Default: all languages returned.
        from_param: A date and optional time for the oldest article allowed. This should be in ISO 8601 format (e.g. **2023-01-22** or **2023-01-22T17:10:27**)

Default: the oldest according to your plan.
        sort_by: The order to sort the articles in. Possible options: **relevancy**, **popularity**, **publishedAt**.
relevancy = articles more closely related to q come first.
popularity = articles from popular sources and publishers come first.
publishedAt = newest articles come first.

Default: **publishedAt**
        to: A date and optional time for the newest article allowed. This should be in ISO 8601 format (e.g. **2023-01-22** or **2023-01-22T17:10:27**)

Default: the newest according to your plan.
        page: Use this to page through the results.

Default: **1**.
        page_size: The number of results to return per page.

Default: **100**. Maximum: **100**.
        
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/google_news/everything"
    querystring = {}
    if domains:
        querystring['domains'] = domains
    if q:
        querystring['q'] = q
    if exclude_domains:
        querystring['exclude_domains'] = exclude_domains
    if language:
        querystring['language'] = language
    if from_param:
        querystring['from_param'] = from_param
    if sort_by:
        querystring['sort_by'] = sort_by
    if to:
        querystring['to'] = to
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_headlines(q: str='messi', category: str='sports', page_size: int=None, page: int=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides live top and breaking headlines for a country, specific category in a country, single source, or multiple sources. You can also search with keywords. Articles are sorted by the earliest date published first.
		
		This endpoint is great for retrieving headlines for use with news tickers or similar."
    q: Keywords or a phrase to search for. 
        category: The category you want to get headlines for. Possible options: **business**, **entertainment**, **general**, **health**, **science**, **sports**, **technology**. Note: you can't mix this param with the sources param. 
        page_size: The number of results to return per page (request). 20 is the default, 100 is the maximum. 
        page: Use this to page through the results if the total results found is greater than the page size. 
        country: The 2-letter ISO 3166-1 code of the country you want to get headlines for. Possible options: **ae**, **ar**, **at**, **au**, **be**, **bg**, **br**, **ca**, **ch**, **cn**, **co**, **cu**, **cz**, **de**, **eg**, **fr**, **gb**, **gr**, **hk**, **hu**, **id**, **ie**, **il**, **in**, **it**, **jp**, **kr**, **lt**, **lv**, **ma**, **mx**, **my**, **ng**, **nl**, **no**, **nz**, **ph**, **pl**, **pt**, **ro**, **rs**, ~~ru~~, **sa**, **se**, **sg**, **si**, **sk**, **th**, **tr**, **tw**, **ua**, **us**, **ve**, **za**. Note: you can't mix this param with the sources param. 
        
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/google_news/top_headlines"
    querystring = {}
    if q:
        querystring['q'] = q
    if category:
        querystring['category'] = category
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_news(player_id: int, slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG and Player_ID
		
		For example: mohamed-salah, 148455"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarkt/news"
    querystring = {'player_id': player_id, 'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_info(slug: str, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG and PLAYER_ID
		
		For example: lionel-messi, 28003"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarkt/player"
    querystring = {'slug': slug, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def squad(club_id: int, slug: str, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG, ID and SEASON
		
		For example: borussia-dortmund, 16, 2022"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarkt/squad"
    querystring = {'club_id': club_id, 'slug': slug, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_profile(user_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter SLUG user.
		For example, from URL  **https://soundcloud.com/siamusic** get only  -     **siamusic**"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/soundcloud_info/{user_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_artist_albums_singles_compilations(id_artist: str, type_search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter Artist ID and type search"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/spotify_discography/{id_artist}/{type_search}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_id_from_name(name_artist: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/search_id_from_name/{name_artist}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_id_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter NAME artist"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/spotify_search_artist/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_realated_artists(id_artist: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter Artist ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/spotify_related_artists/{id_artist}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tunefind_search(artist_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the name of the artist. Please don't use the symbols: #$%^@!/
		For example, in the name "#1 Dads" please remove the symbol # for correct search."
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/tune_find_search/{artist_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventbrite_detail(event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the ID of the festival. Example, luft-2022-tickets-409654235847
		You can get ID from "EventBrite search" API"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/event_brite_detail/{event_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eventbrite_search_event_id(event_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the name of the festival. Example,  LUFT 2022"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/event_brite_search/{event_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songkick_concert(id_conc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Concert info"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/songkick_conc/{id_conc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songkick_artist(artist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Artist info"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/songkick_artist/{artist_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songkick_festivals(id_fest: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "festivals info"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/songkick_fest/{id_fest}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfermarkt_search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by name"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarks/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_artist_concerts(id_artist: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter Artist ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/spotify_artist_concerts/{id_artist}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_artist_overview(id_artist: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter Artist ID"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/spotify_artist_overview/{id_artist}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfermarkt_details(type_s: str, other: str, id_talent: str, part_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter **SLUG ** from Transfermarkt search API.
		
		It os working for all types (players, clubs, managers, referees, etc)"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/transfermarks_detail/{part_slug}/{other}/{type_s}/{id_talent}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songkick_search_artist(artist_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the name of the artist.
		For example, ed sheran"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/songkick_search/{artist_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tunefind_for_details(artist_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Please, enter the slug of artist. For example, taylor-swift or acdc.
		You can get a slug  by using the TuneFind Search API"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/tune_find_details/{artist_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_info_about_artist(parameter: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Give info about artist on allmusic.com
		
		Response -> json"
    
    """
    url = f"https://theclique.p.rapidapi.com/api/v1/get_info_artist/{parameter}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "theclique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

