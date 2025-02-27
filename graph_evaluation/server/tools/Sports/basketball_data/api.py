import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tournament_info(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current season, stage structure(divisions,conferences etc.), country and many more information about a tournament."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/info"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of tournaments in your data coverage."
    
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_teams(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of teams participating in a specific tournament."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/teams"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_results(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including finished matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the matches. The format is {dd/MM/yyyy}. Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/list/results"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_game_leaders(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live points, rebounds and assists leaderboards of the game for triple-double hunting.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/gameleaders"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_statistics(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live in match team statistics for each team in a match.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/statistics"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rebound_leaderboard(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rebounds-per-game leaders of the competition supported with player stats."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/leaderboard/rebound"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_schedule(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team fixtures by all the tournaments a team participates."
    teamid: The id of the team.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/team/schedule"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def assist_leaderboard(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Assists-per-game leaders of the competition supported with player stats."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/leaderboard/assist"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_live(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including live matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the matches. The format is {dd/MM/yyyy}. Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/list/live"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_summary(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Basic summary about the match that includes match scores, match status, team names, venue, and round info.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/summary"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_squad(teamid: int, tournamentid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of players in a team."
    teamid: The id of the team.
        tournamentid: The id of the tournament. Tournament id is optional for club teams. However, it is required for national teams.

        
    """
    url = f"https://basketball-data.p.rapidapi.com/team/squad"
    querystring = {'teamId': teamid, }
    if tournamentid:
        querystring['tournamentId'] = tournamentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_all(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including scheduled, live and finished matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the matches. The format is {dd/MM/yyyy}.  Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/list"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_match_list_scheduled(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily match list including scheduled matches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today - 7 days.**"
    date: The date of the match. The format is {dd/MM/yyyy}. Data can be retrieved for only ± 7 days.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/list/scheduled"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_standings(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team rankings for a specific competition."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/standings"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def point_leaderboard(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Points-per-game leaders of the competition supported with player stats."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/leaderboard/point"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def result_statistics(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the result list of the last 20 matches between the two teams in overall, with home and away filters.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/match/statistics/result/points"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_match_list(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the result list of the last 20 matches between the two teams in overall, with home and away filters.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/match/list/recent"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_fixture(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full match list with quarter, half time and final scores with venue info."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/tournament/fixture"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_statistics_quarter_analysis(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the seasonal quater analysis and statistics of the team in the tournament."
    teamid: The id of the team.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/team/quarter-analysis/seasonal"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_market_statistics_points(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the seasonal market points statistics of the team in the tournament."
    teamid: The id of the team.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/team/statistics/market/seasonal/points"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_margins(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the average match margin statistics of the team in the tournament."
    teamid: The id of the team.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/team/match-margins"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_statistics_points(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the market points statistics which are “First Quarter Result”, “Highest Scoring Quarter” and also “Ordinary/Half Time Winning Margins” for both two teams against the other teams in the tournament.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/match/statistics/market/points"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarter_analysis(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the analyst data for each quarters including Quarter Winning Avereages, Quarter Winning Counts, Quarter Winning Percentages with home team / away team filters.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/match/quarter-analysis"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_advanced_team_statistics(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the season-wide advanced team statistics with home team / away team filters.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/match/statistics/seasonal/teams-advanced"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def under_over_analysis(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the under/over score analysis and statistics with the home/away team filters.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**
		
		P.S.
		Full data is available at Under / Over Analysis endpoint but can not be shown in mock response."
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/match/under-over-analysis"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_boxscore(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live, detailed team and player statistics.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/boxscore"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_play_by_play(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live match events with scores.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Basketball Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/match/playbyplay"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_statistics_points(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the points statistics of the team in the tournament."
    teamid: The id of the team.
        
    """
    url = f"https://basketball-data.p.rapidapi.com/h2h/team/statistics/seasonal/points"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

