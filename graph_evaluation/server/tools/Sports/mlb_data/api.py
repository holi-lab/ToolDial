import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def projected_hitting_stats(player_id: str, league_list_id: str, season: str="'2017'", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players projected hitting stats for a given season.  Omitting the season parameter will return the actual stats for the players earliest major league season."
    player_id: Example: '592789'
        season: Example: '2017'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.proj_pecota_batting.bam"
    querystring = {'player_id': player_id, 'league_list_id': league_list_id, }
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_pitching_stats(season: str, player_id: str, league_list_id: str, game_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players pitching stats for a given season."
    season: Example: '2017'
        player_id: Example: '592789'
        game_type: Example: 'R' The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.sport_pitching_tm.bam"
    querystring = {'season': season, 'player_id': player_id, 'league_list_id': league_list_id, 'game_type': game_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def broadcast_info(src_comment: str, src_type: str, tcid: str, start_date: str="'20170415'", home_away: str="'H'", season: str="'2017'", end_date: str="'20170417'", sort_by: str="'game_time_et_asc'", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information on broadcasts over a given period.  start_date and end_date parameters should be in the YYYYMMDD format.  Although you can omit the home_away parameter to retrieve both home and away game data, one will include the other. For example, a New York Mets home game result will include data for the visiting team."
    start_date: Example: '20170415'
        home_away: Example: 'H' ‘H’ for home games, ‘A’ for away games. Omit for both.
        season: Example: '2017'
        end_date: Example: '20170417'
        sort_by: Example: 'game_time_et_asc' Field to sort results by.
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.mlb_broadcast_info.bam"
    querystring = {'src_comment': src_comment, 'src_type': src_type, 'tcid': tcid, }
    if start_date:
        querystring['start_date'] = start_date
    if home_away:
        querystring['home_away'] = home_away
    if season:
        querystring['season'] = season
    if end_date:
        querystring['end_date'] = end_date
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def career_hitting_stats(player_id: str, game_type: str, league_list_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players career hitting stats for a given game type."
    player_id: Example: '592789'
        game_type: Example: 'R' The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.sport_career_hitting.bam"
    querystring = {'player_id': player_id, 'game_type': game_type, 'league_list_id': league_list_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_hitting_stats(game_type: str, player_id: str, league_list_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players career hitting stats for a given game type split by the league."
    game_type: Example: 'R' The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        player_id: Example: '592789'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.sport_career_hitting_lg.bam"
    querystring = {'game_type': game_type, 'player_id': player_id, 'league_list_id': league_list_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_roster_by_seasons(end_season: str, team_id: str, start_season: str, all_star_sw: str="'N'", sort_order: str='name_asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a teams roster between a given start and end season."
    end_season: Example: '2017'
        team_id: Example: '121'
        start_season: Example: '2016'
        all_star_sw: Example: 'N' Set to ‘Y’ for all star data, and ‘N’ for regular season.
        sort_order: Example: name_asc Field to sort results by.
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.roster_team_alltime.bam"
    querystring = {'end_season': end_season, 'team_id': team_id, 'start_season': start_season, }
    if all_star_sw:
        querystring['all_star_sw'] = all_star_sw
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_pitching_stats(league_list_id: str, game_type: str, player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players career hitting stats for a given game type, split by league."
    game_type: Example: 'R' The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        player_id: Example: '592789'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.sport_career_pitching_lg.bam"
    querystring = {'league_list_id': league_list_id, 'game_type': game_type, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams_by_season(season: str, all_star_sw: str="'N'", sort_order: str='name_asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of major league teams that were active during a given season.  If all_star_sw is set to 'Y', you will instead receive data on the all star teams for that season.  You can sort using the sort_order paramater. Ex: Sort in ascending order by the name field using sort_by='name_asc'"
    season: Example: '2017'
        all_star_sw: Example: 'N' Set to ‘Y’ for all star data, and ‘N’ for regular season.
        sort_order: Example: name_asc Field to sort results by.
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.team_all_season.bam"
    querystring = {'season': season, }
    if all_star_sw:
        querystring['all_star_sw'] = all_star_sw
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def career_pitching_stats(player_id: str, league_list_id: str, game_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players career hitting stats for a given game type."
    player_id: Example: '592789'
        game_type: Example: 'R' The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.sport_career_pitching.bam"
    querystring = {'player_id': player_id, 'league_list_id': league_list_id, 'game_type': game_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_teams(player_id: str, season: str="'2014'", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the teams a player has played for over the course of a season, or their career."
    player_id: Example: '493316'
        season: Example: '2014'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.player_teams.bam"
    querystring = {'player_id': player_id, }
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_search(name_part: str, sport_code: str, active_sw: str="'Y'", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for active and historic/inactive players by name.  The active_sw parameter should be set depending on whether you want to search for active or inactive players. You can omit this parameter, though you will notice a slower response time as the search is done across all (active and inactive) players."
    name_part: Example: 'cespedes%25' The player name to search for.
        active_sw: Example: 'Y' Set to ‘Y’ to search active players, and ‘N’ to search inactive/historic players.
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.search_player_all.bam"
    querystring = {'name_part': name_part, 'sport_code': sport_code, }
    if active_sw:
        querystring['active_sw'] = active_sw
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pitching_leaders(results: str, sort_column: str, season: str, sports_code: str, game_type: str, leader_hitting_repeater_col_in: str='era', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve n leaders for a given hitting statistic.  This endpoint is best used alongside col_in/col_ex to prune response data. Without, it returns entire player objects.  For best results, include the player’s name, id, and the stat as a starting point.  See: Using col_in & col_ex"
    results: Example: 5 The number of results to return.
        sort_column: Example: 'era' The statistic you want leaders for.
        season: Example: '2017'
        game_type: Example: 'R' The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        leader_hitting_repeater_col_in: Example: era
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.leader_pitching_repeater.bam"
    querystring = {'results': results, 'sort_column': sort_column, 'season': season, 'sports_code': sports_code, 'game_type': game_type, }
    if leader_hitting_repeater_col_in:
        querystring['leader_hitting_repeater.col_in'] = leader_hitting_repeater_col_in
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def injuries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all players which are currently injured."
    
    """
    url = f"https://mlb-data.p.rapidapi.com/fantasylookup/json/json/named.wsfb_news_injury.bam"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_transactions_over_period(end_date: str, start_date: str, sport_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all transactions between a given period. start_date and end_date parameters should be in the YYYYMMDD format."
    end_date: Example: '20171231' End date of time period.
        start_date: Example: '20171201' Start date of time period.
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.transaction_all.bam"
    querystring = {'end_date': end_date, 'start_date': start_date, 'sport_code': sport_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_40_man_roster(team_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a team’s 40 man roster."
    team_id: Example: '121'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.roster_40.bam"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hitting_leaders(game_type: str, results: str, sort_column: str, sports_code: str, season: str, leader_hitting_repeater_col_in: str='ab', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve n leaders for a given hitting statistic.  This endpoint is best used alongside col_in/col_ex to prune response data. Without, it returns entire player objects.  For best results, include the player’s name, id, and the stat as a starting point.  See: Using col_in & col_ex"
    game_type: Example: 'R' The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        results: Example: 5 The number of results to return.
        sort_column: Example: 'ab' The statistic you want leaders for.
        season: Example: '2017'
        leader_hitting_repeater_col_in: Example: ab
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.leader_hitting_repeater.bam"
    querystring = {'game_type': game_type, 'results': results, 'sort_column': sort_column, 'sports_code': sports_code, 'season': season, }
    if leader_hitting_repeater_col_in:
        querystring['leader_hitting_repeater.col_in'] = leader_hitting_repeater_col_in
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_type_info(game_type: str, season: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of one or numerous game types.  For example, if you wanted to know when the National League Championship Series was played, this endpoint could tell you that."
    game_type: Example: 'L' 'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        season: Example: '2017'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.org_game_type_date_info.bam"
    querystring = {'game_type': game_type, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def projected_pitching_stats(player_id: str, league_list_id: str, season: str="'2017'", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players projected pitching stats for a given season.  Omitting the season parameter will return the actual stats for the players earliest major league season."
    player_id: Example: '592789'
        season: Example: '2017'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.proj_pecota_pitching.bam"
    querystring = {'player_id': player_id, 'league_list_id': league_list_id, }
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_hitting_stats(league_list_id: str, game_type: str, season: str, player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a players hitting stats for a given season."
    game_type:  Example: 'R'. The type of games you want career stats for.  'R' - Regular Season  'S' - Spring Training  'E' - Exhibition  'A' - All Star Game  'D' - Division Series  'F' - First Round (Wild Card)  'L' - League Championship  'W' - World Series
        season: Example: '2017'
        player_id: Example: '493316'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.sport_hitting_tm.bam"
    querystring = {'league_list_id': league_list_id, 'game_type': game_type, 'season': season, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_info(sport_code: str, player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve general information on a player. This includes name variants, education information, country of origin and attributes like height, weight and age."
    player_id: Example: '493316'
        
    """
    url = f"https://mlb-data.p.rapidapi.com/json/named.player_info.bam"
    querystring = {'sport_code': sport_code, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlb-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

