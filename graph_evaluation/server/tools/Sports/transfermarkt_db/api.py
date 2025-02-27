import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def quick_search(locale: str, query: str, page_number: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quick search. Search players, coaches, agents, clubs, referees by name"
    query: Query
        page_number: Page number
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/search/quick-search"
    querystring = {'locale': locale, 'query': query, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_news(date: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the news by date"
    date: Date news
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/news/list"
    querystring = {'date': date, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_fixtures(club_id: int, season_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club fixtures by club_id"
    club_id: Clubs ID
        season_id: Season ID, endpoint `Seasons of clubs`
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/fixtures"
    querystring = {'club_id': club_id, 'season_id': season_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_comments(locale: str, news_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news comments by news_id"
    news_id: News ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/news/comments"
    querystring = {'locale': locale, 'news_id': news_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def best_players(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the best players. Ex. https://www.transfermarkt.com/statistik/spielertitel?titel_id=195"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/best-players"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_explorer(club_ids: str, locale: str, page_number: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the news by club IDs"
    club_ids: Clubs IDs, separated by commas
        page_number: Page number
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/news/explorer"
    querystring = {'club_ids': club_ids, 'locale': locale, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available_coaches(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available coaches"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/available-coaches"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_value_of_players(sort_by: str, locale: str, position_group: str=None, country_ids: str=None, club_ids: str=None, position_ids: str=None, age_min: int=None, market_value_max: int=None, competition_ids: str=None, age_max: int=None, market_value_min: int=None, page_number: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market value of players"
    position_group: Position group: `GOALKEEPER`, `DEFENDER`, `MIDFIELDER`, `FORWARD`
        country_ids: Country IDs, separated by commas. Ex: 107,124,184,4, use `static/Countries` endpoint
        club_ids: Clubs IDs, separated by commas
        position_ids: Position IDs, separated by commas. Check `static/player-position` endpoint
        age_min: Age min
        market_value_max: Market value max
        competition_ids: Competition IDs, separated by commas. Ex: FS,GB1
        age_max: Age max
        market_value_min: Market value min
        page_number: Page number
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/players-market-value"
    querystring = {'sort_by': sort_by, 'locale': locale, }
    if position_group:
        querystring['position_group'] = position_group
    if country_ids:
        querystring['country_ids'] = country_ids
    if club_ids:
        querystring['club_ids'] = club_ids
    if position_ids:
        querystring['position_ids'] = position_ids
    if age_min:
        querystring['age_min'] = age_min
    if market_value_max:
        querystring['market_value_max'] = market_value_max
    if competition_ids:
        querystring['competition_ids'] = competition_ids
    if age_max:
        querystring['age_max'] = age_max
    if market_value_min:
        querystring['market_value_min'] = market_value_min
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_trend(club_ids: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the club trend by club_ids"
    club_ids: Clubs IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/trend"
    querystring = {'club_ids': club_ids, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs_short_info(club_ids: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short info about clubs by IDs"
    club_ids: Clubs IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/short-info"
    querystring = {'club_ids': club_ids, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_competition_statistics(locale: str, country_id: int=189, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the transfers competition statistics"
    country_id: Country ID. Use `static/Countries` endpoint
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/competition-statistics"
    querystring = {'locale': locale, }
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_details(locale: str, news_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news details by news_id"
    news_id: News ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/news/details"
    querystring = {'locale': locale, 'news_id': news_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_valuable_teams(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the most valuable teams"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/most-valuable-teams"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_rankings(club_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club rankings by club_id"
    club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/rankings"
    querystring = {'club_id': club_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_transfers(season_id: int, locale: str, club_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club transfers by club_id. List of incoming and outgoing transfers"
    season_id: Season ID, endpoint `Seasons of clubs`
        club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/transfers"
    querystring = {'season_id': season_id, 'locale': locale, 'club_id': club_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def longest_tenure_as_a_coach(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the longest tenure as a coach"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/longest-tenure-as-coach"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_of_competition(locale: str, competition_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition seasons"
    competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/seasons"
    querystring = {'locale': locale, 'competition_id': competition_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_national_competitions(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of National competitions"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/national-competitions"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fifa_rankings(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "These are the most recent FIFA world rankings. Ex. https://www.transfermarkt.com/statistik/weltrangliste"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/rankings/fifa"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_play_day_matches(locale: str, competition_id: str, season_id: int, match_day: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition play day matches"
    competition_id: Competition ID
        season_id: Season ID, endpoint `Seasons of competition`
        match_day: Match day
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/play-day-matches"
    querystring = {'locale': locale, 'competition_id': competition_id, 'season_id': season_id, 'match_day': match_day, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_most_valuable_teams(competition_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of most valuable teams by competition_id"
    competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/most-valuable-teams"
    querystring = {'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_performance(locale: str, season_id: int, competition_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get performance of the competition players"
    season_id: Season ID, endpoint `Seasons of competition`
        competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/players-performance"
    querystring = {'locale': locale, 'season_id': season_id, 'competition_id': competition_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staff_profile(staff_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get staff (coaches) profile by IDs"
    staff_id: Staff ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/staff/profile"
    querystring = {'staff_id': staff_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def short_info(locale: str, staff_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short info about staff (coaches) by IDs"
    staff_ids: Staff IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/staff/short-info"
    querystring = {'locale': locale, 'staff_ids': staff_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_info(competition_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition info"
    competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/info"
    querystring = {'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_top_cup_competitions(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of Top cup competitions by locale"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/top-cup-competitions"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_short_info(locale: str, competition_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competitions short info by competition IDs"
    competition_ids: Competition IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/short-info"
    querystring = {'locale': locale, 'competition_ids': competition_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_club_competitions(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of Club competitions"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/club-competitions"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_game_plan(locale: str, season_id: int, competition_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition game plan"
    season_id: Season ID, endpoint `Seasons of competition`
        competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/game-plan"
    querystring = {'locale': locale, 'season_id': season_id, 'competition_id': competition_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staff_achievements(locale: str, staff_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get staff (coaches) achievements by IDs"
    staff_id: Staff ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/staff/achievements"
    querystring = {'locale': locale, 'staff_id': staff_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rumor_details(rumor_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rumor details by rumor_id"
    rumor_id: Rumor ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/rumors/details"
    querystring = {'rumor_id': rumor_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_top_competitions(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of Top competitions by locale"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/top-competitions"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_transfers(locale: str, position_group: str=None, market_value_min: int=None, club_id: int=None, competition_id: str=None, position_id: int=None, page_number: int=0, top_transfers_first: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of transfers"
    position_group: Position group: `GOALKEEPER`, `DEFENDER`, `MIDFIELDER`, `FORWARD`
        market_value_min: Market value min
        club_id: Clubs ID
        competition_id: Competition ID
        position_id: Position ID, Check `static/player-position` endpoint
        page_number: Page number
        top_transfers_first: Only top transfers
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/transfers/list"
    querystring = {'locale': locale, }
    if position_group:
        querystring['position_group'] = position_group
    if market_value_min:
        querystring['market_value_min'] = market_value_min
    if club_id:
        querystring['club_id'] = club_id
    if competition_id:
        querystring['competition_id'] = competition_id
    if position_id:
        querystring['position_id'] = position_id
    if page_number:
        querystring['page_number'] = page_number
    if top_transfers_first:
        querystring['top_transfers_first'] = top_transfers_first
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_report(locale: str, fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get report by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/report"
    querystring = {'locale': locale, 'fixture_id': fixture_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_top_10(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of top 10 players"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/top10"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_list(locale: str, timezone_offset: int, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of fixtures by date. Use `Club Short info` to get team names"
    timezone_offset: Timezone offset
        date: Date
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/list"
    querystring = {'locale': locale, 'timezone_offset': timezone_offset, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_highlights(fixture_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture highlights by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/highlights"
    querystring = {'fixture_id': fixture_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_result(locale: str, fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture result by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/result"
    querystring = {'locale': locale, 'fixture_id': fixture_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_info(locale: str, fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture info by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/info"
    querystring = {'locale': locale, 'fixture_id': fixture_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_statistics(fixture_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture statistics by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/statistics"
    querystring = {'fixture_id': fixture_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_events(locale: str, fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture events by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/events"
    querystring = {'locale': locale, 'fixture_id': fixture_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_rumors(sort_by: str, locale: str, club_ids: str=None, include_closed: bool=None, market_value_max: int=None, player_ids: str=None, allow_secondary_positions: bool=None, position_id: int=None, position_group: str=None, competition_ids: str=None, market_value_min: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of rumors"
    club_ids: Clubs IDs, separated by commas
        include_closed: Include closed rumors
        market_value_max: Market value max
        player_ids: Players IDs, separated by commas
        allow_secondary_positions: Allow secondary positions
        position_id: Position ID. Check `static/player-position` endpoint
        position_group: Position group: `GOALKEEPER`, `DEFENDER`, `MIDFIELDER`, `FORWARD`
        competition_ids: Competition IDs, separated by commas. Ex: FS,GB1
        market_value_min: Market value min
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/rumors/list"
    querystring = {'sort_by': sort_by, 'locale': locale, }
    if club_ids:
        querystring['club_ids'] = club_ids
    if include_closed:
        querystring['include_closed'] = include_closed
    if market_value_max:
        querystring['market_value_max'] = market_value_max
    if player_ids:
        querystring['player_ids'] = player_ids
    if allow_secondary_positions:
        querystring['allow_secondary_positions'] = allow_secondary_positions
    if position_id:
        querystring['position_id'] = position_id
    if position_group:
        querystring['position_group'] = position_group
    if competition_ids:
        querystring['competition_ids'] = competition_ids
    if market_value_min:
        querystring['market_value_min'] = market_value_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_lineups(fixture_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture lineups by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/lineups"
    querystring = {'fixture_id': fixture_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_ticker(locale: str, fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture ticker by fixture_id, live-ticker"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/ticker"
    querystring = {'locale': locale, 'fixture_id': fixture_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_standings(locale: str, fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixture standings by fixture_id"
    fixture_id: Fixture ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/fixtures/standings"
    querystring = {'locale': locale, 'fixture_id': fixture_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_competitions(locale: str, country_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get country competitions by countries ID"
    country_id: Country ID. Use `static/Countries` endpoint
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/countries/competitions"
    querystring = {'locale': locale, 'country_id': country_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_short_info(locale: str, country_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short info about countries by IDs"
    country_ids: Country IDs, separated by commas. Use `static/Countries` endpoint
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/countries/short-info"
    querystring = {'locale': locale, 'country_ids': country_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_cups(locale: str, country_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get country cups by countries ID"
    country_id: Country ID. Use `static/Countries` endpoint
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/countries/cups"
    querystring = {'locale': locale, 'country_id': country_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of countries"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/static/countries"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def static_values(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get static values"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/static/values"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clean_sheets(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of clean sheets players. Ex. https://www.transfermarkt.com/statistik/weisseweste"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/clean-sheets"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_valuable_players(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the most valuable players, eleven players (virtual team)"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/most-valuable-players"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs_upcoming_fixtures(club_ids: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get upcoming fixtures by club_ids"
    club_ids: Clubs IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/upcoming-fixtures"
    querystring = {'club_ids': club_ids, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_of_clubs(locale: str, club_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short seasons of clubs by club_id"
    club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/seasons"
    querystring = {'locale': locale, 'club_id': club_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_info(locale: str, club_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club info by club_id"
    club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/info"
    querystring = {'locale': locale, 'club_id': club_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_valuable_competitions(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the most valuable competitions"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/most-valuable-competitions"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_squad(season_id: int, locale: str, club_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club squad by club_id"
    season_id: Season ID, endpoint `Seasons of clubs`
        club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/squad"
    querystring = {'season_id': season_id, 'locale': locale, 'club_id': club_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def golden_boot(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of golden boot (shoe) players. Ex. https://www.transfermarkt.com/statistik/goldenerschuh"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/golden-boot"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_profile(locale: str, club_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club profile by club_id. Main facts, stadium, additional teams"
    club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/profile"
    querystring = {'locale': locale, 'club_id': club_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_records(locale: str, page_number: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the transfers records"
    page_number: Page number
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/markets/transfers-records"
    querystring = {'locale': locale, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compare_players(player_ids: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compare players by player_ids"
    player_ids: Players IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/compare"
    querystring = {'player_ids': player_ids, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_profile(player_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player profile. Name, image, shirt number, nationalities, market value, club, age, foot, height"
    player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/profile"
    querystring = {'player_id': player_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_market_value_progress(locale: str, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player market value"
    player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/market-value-progress"
    querystring = {'locale': locale, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_performance(season_id: int, player_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player performance"
    season_id: Season ID, endpoint `Seasons of competition`
        player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/performance"
    querystring = {'season_id': season_id, 'player_id': player_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_images(locale: str, player_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player images by player_ids"
    player_ids: Players IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/images"
    querystring = {'locale': locale, 'player_ids': player_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_transfers(player_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player transfers by player_id"
    player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/transfers"
    querystring = {'player_id': player_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compare_clubs(locale: str, club_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compare clubs by club_ids, only 2 clubs"
    club_ids: Clubs IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/compare"
    querystring = {'locale': locale, 'club_ids': club_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_short_info(player_ids: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get short info about players by IDs"
    player_ids: Players IDs, separated by commas
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/short-info"
    querystring = {'player_ids': player_ids, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_staff(club_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get club staff by club_id. List of staff"
    club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/staff"
    querystring = {'club_id': club_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_news(club_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest club news by club_id"
    club_id: Clubs ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/clubs/news"
    querystring = {'club_id': club_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_injuries(locale: str, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player injuries. Date and description"
    player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/injuries"
    querystring = {'locale': locale, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_news(locale: str, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player news"
    player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/news"
    querystring = {'locale': locale, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_progress(player_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player progress"
    player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/progress"
    querystring = {'player_id': player_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def european_champions(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "European champions. Titles and success"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/rankings/european-champions"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uefa_rankings(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get countries within 5 years. Ex. https://www.transfermarkt.com/statistik/5jahreswertung"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/rankings/uefa"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_champions(competition_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of champions"
    competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/champions"
    querystring = {'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_clubs(competition_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of clubs"
    competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/clubs"
    querystring = {'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_performance_details(competition_id: str, season_id: int, locale: str, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player performance details"
    competition_id: Competition ID
        season_id: Season ID, endpoint `Seasons of competition`
        player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/performance-details"
    querystring = {'competition_id': competition_id, 'season_id': season_id, 'locale': locale, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referee_profile(referee_id: int, locale: str, season_id: int=2021, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get referee profile by referee_id"
    referee_id: Referee ID
        season_id: Season ID, endpoint `Seasons of competition`
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/referees/profile"
    querystring = {'referee_id': referee_id, 'locale': locale, }
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_standings(standing_type: str, competition_id: str, season_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition standings"
    competition_id: Competition ID
        season_id: Season ID, endpoint `Seasons of competition`
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/standings"
    querystring = {'standing_type': standing_type, 'competition_id': competition_id, 'season_id': season_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_info(player_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player info. Name, image, shirt number, nationalities, market value, club"
    player_id: Player ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/players/info"
    querystring = {'player_id': player_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def world_cup_champions(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World cup champions. Titles and success"
    
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/rankings/world-cup-champions"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_coaches(locale: str, competition_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of coaches. Use endpoint `Staff Short info` to get more information by coach ID"
    competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/coaches"
    querystring = {'locale': locale, 'competition_id': competition_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_news(locale: str, competition_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news about competition"
    competition_id: Competition ID
        
    """
    url = f"https://transfermarkt-db.p.rapidapi.com/v1/competitions/news"
    querystring = {'locale': locale, 'competition_id': competition_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transfermarkt-db.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

