import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def red_card_leaderboard(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of players shown most red cards of the competition supported with player stats."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/leaderboard/red"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
    url = f"https://soccer-data.p.rapidapi.com/team/squad"
    querystring = {'teamId': teamid, }
    if tournamentid:
        querystring['tournamentId'] = tournamentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
    date: Date of the match. The format is {dd/MM/yyyy}.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/list/results"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
    date: Date of the match. The format is {dd/MM/yyyy}. 
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/list"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yellow_card_leaderboard(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of players shown most yellow cards of the competition supported with player stats."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/leaderboard/yellow"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
    url = f"https://soccer-data.p.rapidapi.com/tournament/teams"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_info(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current season, stage structure(groups,knockout stages etc.), country and many more information about a tournament."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/info"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
    date: Date of the match. The format is {dd/MM/yyyy}.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/list/scheduled"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goal_leaderboard(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top scorers of the competition supported with player stats."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/leaderboard/goal"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_events(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Goals, yellow cards, red cards and substitutions in a match.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/event/basic"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_standings(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team Rankings for a specific competition."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/standings"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_squad_with_stats(teamid: int, tournamentid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of players in a team with their respective stats."
    teamid: The id of the team.
        tournamentid: The id of the tournament. Tournament id is optional for club teams. However, it is required  for National Teams.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/team/squad/stats"
    querystring = {'teamId': teamid, }
    if tournamentid:
        querystring['tournamentId'] = tournamentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
    url = f"https://soccer-data.p.rapidapi.com/team/schedule"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_events_all(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All of the events occuring in a match.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/event/all"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
    date: Date of the match. The format is {dd/MM/yyyy}.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/list/live"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_commentary(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live event based commentary with texts and events.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/commentary"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def missing_player_list(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of injured and suspended players for the matches in a specific date.
		
		**The data will return for only -+7 days period, so endpoint can be tested with date range of today +- 7 days.**"
    date: Date of the matches. The format is 	{dd/MM/yyyy}.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/missing-players-list"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_lineup(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team lineups, formations and head coaches.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/lineup"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/statistics"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_summary(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Match scores, match status, team names, stadium, referee and weather forecast.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/match/summary"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_market_statistics_goals(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides under/over, both team to score and also total goals statistics of the team in the tournament.
		
		P.S.
		Full data is available at Seasonal Market Statistics: Goals endpoint but can not be shown in mock response."
    teamid: The id of the team.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/team/statistics/market/seasonal/goals"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_tables_under_over(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides under/over statistics of the teams in the tournament, ordered by the tournament standings.
		
		P.S.
		Full data is available at Market Tables: Under/Over endpoint but can not be shown in mock response."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/tournament/standings/under-over"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_tables_total_goals(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides total goals statistics of the teams in the tournament, ordered by the tournament standings.
		
		P.S.
		Full data is available at Market Tables: Total Goals endpoint but can not be shown in mock response."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/tournament/standings/total-goals"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_tables_both_teams_to_score(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides both team to score statistics of the teams in the tournament, ordered by the tournament standings.
		
		P.S.
		Full data is available at Market Tables: Both Teams to Score endpoint but can not be shown in mock response."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/tournament/standings/both-teams-to-score"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referee_statistics(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the information about referee and his/her statistics.
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/referee"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goal_analysis_scoring_first_in_between(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the scoring first goal statistics of the teams between in the minutes intervals(0-10, 11-20, 21-30...etc.)
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/goal-analysis/scoring-first-between"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_goal_analysis_goal_conceded_minutes(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the goal conceded goal statistics of the teams against to other teams in the tournament in the minutes intervals(0-10, 11-20, 21-30...etc.)"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/goal-analysis/goal-conceded-minutes/seasonal"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_goal_analysis_goal_minutes(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the goals statistics of the teams against to the other teams(in tournament) in the minutes intervals(0-10, 11-20, 21-30...etc.)
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/goal-analysis/goal-minutes/seasonal"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.
		
		P.S.
		Full data is available at Recent Match List endpoint but can not be shown in mock response."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/list/recent"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_statistics_goals(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the market goals statistics which are “Both Teams to Score”, “Under/Over” and also “Total Goals” for both two teams against the other teams in the tournament.
		
		P.S.
		Full data is available at Market Statistics: Goals endpoint but can not be shown in mock response.
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/statistics/market/goals"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_statistics_goals(teamid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the goal statistics of the team in the tournament."
    teamid: The id of the team.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/team/statistics/seasonal/goals"
    querystring = {'teamId': teamid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_goal_analysis_conceding_first(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the conceding first goal statistics of the teams against to other teams in the tournament in the minutes intervals(0-10, 11-20, 21-30...etc.)
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/goal-analysis/conceding-first/seasonal"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
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
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/statistics/result/goals"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_streaks(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the streaks(winning, draw, losing...etc.) of both two teams in tournament.
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/streaks/seasonal"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goal_analysis_goal_minutes_in_between(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the goals statistics of the teams between in the minutes intervals(0-10, 11-20, 21-30...etc.)
		
		The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints."
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/goal-analysis/goal-minutes-between"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonal_goal_analysis_scoring_first(matchid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the scoring first goal statistics of the teams against to other teams in tournaments in the minutes intervals(0-10, 11-20, 21-30...etc.)
		
		**The data will return for only -+7 days period, so endpoint can be tested with match that its date is in today +- 7 days. You can choose suitable match from Soccer Match List or Fixture endpoints.**"
    matchid: The id of the match.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/h2h/match/goal-analysis/scoring-first/seasonal"
    querystring = {'matchId': matchid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides list of tournaments."
    
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_fixture(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Full match list with half time and final scores, with additional info for each match such as referee and stadium."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/fixture"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def assist_leaderboard(tournamentid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides assist leaderboard data of the tournament."
    tournamentid: The id of the tournament.
        
    """
    url = f"https://soccer-data.p.rapidapi.com/tournament/leaderboard/assist"
    querystring = {'tournamentId': tournamentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

