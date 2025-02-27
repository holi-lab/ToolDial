import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def arbitrage_low_hold(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns the bets that have arbitrage and all bets that have low holds. It may be a bit confusing at the moment so if you have questions feel free to ask. Just as a basic overview, all of the lines for each side of the bet are shown in "outcomes", in "alt_low_hold" it shows the combinations of 2 sites that make up low hold bets from those outcomes, in "alt_arb" it likewise shows the combinations that are arbitrage."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/arb_low_hold"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_odds_by_category_schedule(category: str, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you query the category schedule and do not include a "sport" and "category" parameter it will return all possible values of sport and their respective categories. If you do include those parameters it will return a schedule of all sites for that sport and category. Remember that the categories are hashes, 0 is moneylines, 1 is spreads, 2 is over under, etc. There are a lot of categories, refer to the deconstruct ID code to determine which is which."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/category_schedule"
    querystring = {'category': category, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_odds_basic_lines_schedule(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a schedule that shows moneylines, spreads, and over under offerings by all bookmakers."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/basic_schedule"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_list(league_name: str='MLB', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Valid League Names: "NBA", "NCAA Basketball", "NFL", "NCAA Football", "NHL", "MLB"
		
		This will return a dictionary of games where the keys will be hashes describing the game and the values will be dictionaries of all of the games attributes including which sites have that game for betting and when our data last updated that site.
		
		If you want all of the games for all of the leagues just leave the "league_name" parameter empty."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/games_list"
    querystring = {}
    if league_name:
        querystring['league_name'] = league_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_odds_flat_schedule(event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a flat schedule of lines where the keys of 
		the dictionary are tags that fully describe the bet. 
		There is a program in the documentation called 
		decode_hash.py that will break these hashes down into 
		descriptions.
		
		If the bet has both an over and an under then the 
		vf_odds_pct will represent the implied win percent of 
		this bet from the vig free line.
		
		Structure:
		{
		  "bet_hash Eg. PP^0^(Jaylen Brown)^25.5*O": {
		    "Site1 Eg. Fanduel": {
		      "site": "Fanduel",
		      "line": 25.5,
		      "dec_odds": 1.87,
		      "full_hash": "20001x20004@2023-02-09T00_PP^0^(Jaylen Brown)^25.5*O_FanDuel",
		      "am_odds": "-115",
		      "open": true,
		      "vf_odds_pct": 0.50411
		    }
		  }
		}"
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/flat_schedule"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def low_hold_bets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns the bets that have low hold bets. It may be a bit confusing at the moment so if you have questions feel free to ask. Just as a basic overview, all of the lines for each side of the bet are shown in "outcomes", in "alt_low_hold" it shows the combinations of 2 sites that make up low hold bets from those outcomes, in "alt_arb" it likewise shows the combinations that are arbitrage."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/low_holds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def arbitrage_bets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns the bets that have arbitrage. It may be a bit confusing at the moment so if you have questions feel free to ask. Just as a basic overview, all of the lines for each side of the bet are shown in "outcomes", in "alt_low_hold" it shows the combinations of 2 sites that make up low hold bets from those outcomes, in "alt_arb" it likewise shows the combinations that are arbitrage."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/arbs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_odds_tiered_schedule(event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a schedule that is tiered and separated with qualitative information. It is used for the website BookBreakers.us to create accordion tables of bet data.
		
		The Tiers:
		1. Bet Type and Game Period (Eg. P3^0 is 3 Pointers Full Game in Basketball)
		2. Player (Eg. Jayson Tatum) (NOTE: If this is not a prop this tier will not exists)
		3. Lines Available (Eg. 2.5, 3.5)
		4. Outcomes for Those Lines (Eg. O, U for Over and Under)
		5. Sites (Eg. Fanduel, DraftKings, ETC.)"
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/tiered_schedule"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def line_updates_changes_in_odds(since_timestamp: str='1676477789', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All new odds and changes to odds from all sites are compiled in these dictionaries. The dictionaries are timestamped for exactly when they were processed and the exact update times in UTC of the lines are shown in the dictionary (labeled "ut" for "update time"). Use GET param "since_timestamp" as a UTC timestamp to get updates since a given time. If you do not include this parameter it give only the most recent updates file. 3 minutes is the maximum amount of updates stored by the server."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/line_updates"
    querystring = {}
    if since_timestamp:
        querystring['since_timestamp'] = since_timestamp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_odds_by_site_schedule(site: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a schedule that is separated by site. Upper and lower case doesn't matter as long as it has the correct letters for the request (DraftKings is same as draftkings)
		
		The Sites:
		1. Fanduel
		2. DraftKings
		3. Caesars
		4. WynnBet
		5. PointsBet
		6. BetMGM
		7. SuperBook
		8. FoxBet
		9. Kambis    (SugarHouse/BetRivers/Barstool)"
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/site_schedule"
    querystring = {'site': site, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookbreakers_team_lookups(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return our dictionary of team names and aliases that different sites use for this game. Our matching games algorithm uses this to determine which team names on different sites correspond to the team name on ESPN's schedule of games."
    
    """
    url = f"https://sportsbook-odds.p.rapidapi.com/team_lookups"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsbook-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

