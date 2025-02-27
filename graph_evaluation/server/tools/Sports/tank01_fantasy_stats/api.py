import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_news_and_headlines(topnews: str=None, playerid: str=None, maxitems: str='10', recentnews: str='true', fantasynews: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve relevant news links.
		/getNBANews
		
		Options:
		- playerID=xxx  
		Enter a playerID here and get news for that player
		
		- topNews=true  
		Retrieves top news
		
		- fantasyNews=true  
		Retrieves top fantasy-relevant news 
		
		- recentNews=true  
		Retrieves most recent news, whether it's really fantasy relevant or not
		
		- maxItems=xx  
		Enter a number there to limit the number of responses you'll receive back.
		
		All news will have a "link" and "title" element. Some will have "image" and some will have playerID (single item) or playerIDs (list)."
    topnews: true
        playerid: playerID can be found in many of the endpoints
        maxitems: number
        recentnews: true
        fantasynews: true
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBANews"
    querystring = {}
    if topnews:
        querystring['topNews'] = topnews
    if playerid:
        querystring['playerID'] = playerid
    if maxitems:
        querystring['maxItems'] = maxitems
    if recentnews:
        querystring['recentNews'] = recentnews
    if fantasynews:
        querystring['fantasyNews'] = fantasynews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nba_games_and_stats_for_a_single_player(playerid: str, numberofgames: str=None, season: str='2023', gameid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call will grab a map of all of the games a player has played in the season.     
		playerID is a required parameter.
		You can also use gameID if you want to only pull back a specific game. 
		season is an optional parameter.  Currently only 2022 (last season) and 2023 (this season) are available.  If you do not include season as a parameter, it will return this season's games. 
		
		Example:
		Correct way to get the stats for Nikola Vucevic for his game against the Heat on 10/19/2022 would be this:
		/getNBAGamesForPlayer?playerID=28268405032&gameID=20221019_CHI@MIA
		
		But if you wanted to get all of his games this season, you'd make this call
		/getNBAGamesForPlayer?playerID=28268405032
		
		If you want his games from last season, do this..
		/getNBAGamesForPlayer?playerID=28268405032&season=2022
		
		But if you wanted to get all of his games this season, you don't have to include the season. Simply make this call
		/getNBAGamesForPlayer?playerID=28268405032
		
		You can limit the amount of games returned with parameter: numberOfGames.   For example: &numberOfGames=5 will return the last 5 games this player has an entry for.
		
		This call will not work without playerID.  If you want stats for all players during a game, then use the getNBABoxScore call with that specific gameID."
    playerid: playerID required
        numberofgames: use this to restrict the results to the most recent numberOfGames you give here for the parameter
        season: season in format YYYY
defaults to current season if gameID is not listed
        gameid: use this to restrict the results to the boxscore for only one game of this player.  This increases the speed of the call.
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
    querystring = {'playerID': playerid, }
    if numberofgames:
        querystring['numberOfGames'] = numberofgames
    if season:
        querystring['season'] = season
    if gameid:
        querystring['gameID'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams(statstoget: str='averages', schedules: str='true', teamstats: str='true', topperformers: str='true', rosters: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call will retrieve the list of NBA teams. Included is their name, city, abbreviation, and teamID which can be used in other calls.   This also includes standings (win/loss/ppg/oppg/streak) data.
		
		In the team rosters, all player information is available, which includes current injury status. 
		
		/getNBATeams
		
		Optional parameters are schedules=true ,  rosters=true , and/or topPerformers=true
		
		topPerformers get returned in a list for each stat, since a team might have multiple leaders averaging the same amount of whichever stat."
    schedules: schedules=\\\"true\\\"  to add team schedules to the data returned
        teamstats: teamStats=true to add team stats to the data returned
        topperformers: topPerformers=true  to add the team's stat leaders to the data returned
        rosters: rosters=\\\"true\\\" to add team rosters to the data returned
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBATeams"
    querystring = {}
    if statstoget:
        querystring['statsToGet'] = statstoget
    if schedules:
        querystring['schedules'] = schedules
    if teamstats:
        querystring['teamStats'] = teamstats
    if topperformers:
        querystring['topPerformers'] = topperformers
    if rosters:
        querystring['rosters'] = rosters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nba_betting_odds(gameid: str=None, gamedate: str='20230304', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This grabs NBA betting/gambling lines and odds from some of the most popular online sportsbooks (fanduel, betrivers, betmgm, caesars, pointsbet, etc). 
		
		You can call this for specific game or a specific date.  Check out the example responses here for the type of data you can expect back.   Some of the sportsbooks do not offer live betting, so data from those sportsbooks will not be returned after the game starts.  
		
		
		Either gameDate or gameID is required.
		Examples of what the calls can look like:
		/getNBABettingLines?gameDate=20221227
		/getNBABettingLines?gameID=20221227_DEN@SAC"
    gamedate: format YYYYMMDD
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBABettingOdds"
    querystring = {}
    if gameid:
        querystring['gameID'] = gameid
    if gamedate:
        querystring['gameDate'] = gamedate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_player_information(statstoget: str='averages', playerid: str=None, playername: str='smith', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call this to get general information on each player (name, team, experience, birthday, college, etc).
		
		You can call with their playerID, if you know it.  playerID calls will always be quicker as it acts on the key of the table.  
		
		You can also call with playerName.  This call will return a list of players who have that name.  If you want to include spaces in the search name, then use underscore.  So if you want to find LeBron, you can use lebron_james and it will bring him back.  Or try with playerName=smith and it will return a list of guys with smith in their name.
		
		/getNBAPlayerInfo?playerID=28908111729
		
		/getNBAPlayerInfo?playerName=smith
		
		etc"
    statstoget: can be: totals or averages
Works for current season only.
        playerid: Numerical playerID
either playerID or playerName is required
If playerID is used then the endpoint returns a map, not a list, as only one player will be returned.
        playername: player name
either playerID or playerName is required
If this is used, the endpoint returns a list of players that match the entered playerName.
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo"
    querystring = {}
    if statstoget:
        querystring['statsToGet'] = statstoget
    if playerid:
        querystring['playerID'] = playerid
    if playername:
        querystring['playerName'] = playername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nba_injury_list_history(year: str=None, endinjdate: str=None, numberofdays: str=None, beginninginjdate: str=None, playerid: str=None, injdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This table currently has injury history from the years 2020, 2021, 2022, and 2023 for players who played an NBA game in the 21-22 season and afterwards.  Consider this table as "beta" at this point, as we are still collecting previous seasons' data and working through some challenges. 
		
		parameters:
		(All dates must be in format YYYYMMDD)
		playerID - Use this if you only want to pull back injury history from a specific player.
		injDate - Use this is to pull back injury history for only a specific date.
		The next two are used to pull back a range of dates.  You can use one or the other. Or none. 
		beginningInjDate - Lower boundary of the range of dates. Inclusive
		endInjDate - Upper boundary of the range of dates.  Inclusive.
		year - If your range of dates include multiple years, the api will only pull back dates for one year.  If year isn't selected then you will get injuries from current year.   Using "year" without any other parameters will do nothing and the api will still bring back the default, last 14 days of injuries.
		numberOfDays - Valid for numbers 0 through 30, you can pull back information from the previous 0 to 30 days.  
		Calling this endpoint with no parameters will give a list of all players injuries from the last 14 calendar days. 
		
		Again, please consider this endpoint as not fully functional and it could be buggy.  We'll update the description here whenever we add more years, add options, or finalize the interface.
		
		We encourage testing and feedback here.  Thanks in advance!"
    year: format YYYY
Use this to restrict results to one year
        endinjdate: format YYYYMMDD
Use this to restrict date range. This date is inclusive and the end of the date range.
        numberofdays: Number of days should be from 1 to 30, and will restrict your results to the most recent number of days you provide as the variable.
        beginninginjdate: format YYYYMMDD
Use this to restrict date range. This date is inclusive and the beginning of the date range.
        playerid: Numerical playerID
Restricts results to a specific player only.
        injdate: format YYYYMMDD
Use this if you want to restrict your results to only one date
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAInjuryList"
    querystring = {}
    if year:
        querystring['year'] = year
    if endinjdate:
        querystring['endInjDate'] = endinjdate
    if numberofdays:
        querystring['numberOfDays'] = numberofdays
    if beginninginjdate:
        querystring['beginningInjDate'] = beginninginjdate
    if playerid:
        querystring['playerID'] = playerid
    if injdate:
        querystring['injDate'] = injdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_roster(teamid: str=None, statstoget: str='averages', archivedate: str=None, teamabv: str='SAC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns the current or historical* roster of any team, using the teamID that can be found in "getNBATeams" call.
		
		Rosters are updated a few times throughout the day.  Usually once per hour and before each game starts.
		 
		Call needs to look like this:
		/getNBATeamRoster?teamID=1
		or
		/getNBATeamRoster?teamAbv=ATL
		
		
		That will return a list of the team's current roster players in the body.
		
		Historical rosters are saved on a daily basis as of 20230505 and moving forward. 
		
		Add parameter archiveDate to the call to get a list of roster players (playerID's only) for that specific date.  Historical roster dates only are kept as far back as 20230505."
    teamid: Number 1 - 30
        statstoget: either: totals or averages
Does not work with archiveDate
        archivedate: format YYYYMMDD
        teamabv: format CHI, BOS, ATL, etc
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster"
    querystring = {}
    if teamid:
        querystring['teamID'] = teamid
    if statstoget:
        querystring['statsToGet'] = statstoget
    if archivedate:
        querystring['archiveDate'] = archivedate
    if teamabv:
        querystring['teamAbv'] = teamabv
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_schedule(season: str='2023', teamid: str=None, teamabv: str='GS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns the schedule of any team, using the teamID that can be found in "getNBATeams" call.
		
		Call needs to look like this:
		
		/getNBATeamSchedule?teamID=1
		
		You can also use the team Abbreviation:
		
		/getNBATeamSchedule?teamAbv=CHI   
		
		That will return a list of the team's games in the body. 
		
		To get a list of appropriate team abbreviations, use the getTeams call.
		
		You can also add the "season" parameter if you want to specify season.  Right now we only have season 2022 and 2023.   
		Default season is current season."
    season: format YYYY and defaults to current season's year.  For instance, NBA season 2022-23 would just be formatted 2023. 
this only works for 2022 and 2023
        teamid: teamID is 1-30
        teamabv: teamAbv is format like CHI, BOS, or ATL, etc
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamSchedule"
    querystring = {}
    if season:
        querystring['season'] = season
    if teamid:
        querystring['teamID'] = teamid
    if teamabv:
        querystring['teamAbv'] = teamabv
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_player_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grabs all of this season's players and their IDs.  
		
		ONE CALL is all you need to retrieve every player.  No need to call multiple times to get the full list.
		
		Rosters are updated multiple times per day during the season.
		
		You mainly will use this to match a player with his playerID.
		
		There are no parameters, just a simple call..
		
		/getNBAPlayerList"
    
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_schedule(gamedate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get basic information on which games are being played during a day.  
		call is like this:
		/getNBAGamesForDate?gameDate=20220310
		The above call will return all of the games from March 10th, 2022.  Date must be in that format.  
		For March 10th, there were two games. They come back in a list format within the body of the response."
    gamedate: format: YYYYMMDD
required
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate"
    querystring = {'gameDate': gamedate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_general_game_information(gameid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call pulls back the most general information for a game: away team, home team, game date, and game start time.  All times are in Eastern (ET) time zone.  
		gameID is needed.  You can retrieve gameID from a few different calls.  The "getNBAGamesForDate" call or the "getNBATeamSchedule" call will be the best ways to get the gameID's.   
		
		Call should look like this: 
		/getNBAGameInfo?gameID=20220310_BKN@PHI"
    
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAGameInfo"
    querystring = {'gameID': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_game_box_score_live_real_time(gameid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the entire box score for a game either in progress or already completed for the current season.  The stats retrieved here are what are normally shown in box scores or used in fantasy games.  If there are any stats here that you'd like to see, please message me.  
		
		This does retrieve live stats, however please keep in mind that NBA games tend to begin about 10 minutes later than their official start time.  This can be even longer for nationally televised games.  So if you are looking for live stats of a game that starts at 7, and you don't see anything even at 7:10, don't worry.  It will show up almost immediately after the game starts.    
		
		The call looks like this /getNBABoxScore?gameID=20220310_BKN@PHI
		
		The call needs to be exactly in the same format as above.  8 digit date, underscore, then the away team abbreviation, @, then home team abbreviation.  Complete list of team abbreviations can be retrieved with the getNBATeams call or various other calls.  
		
		But, the best way to find specific game ID's are either from the "getNBAGamesForDate" call, or the "getNBATeamSchedule" call."
    
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBABoxScore"
    querystring = {'gameID': gameid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_scoreboard_live_real_time(topperformers: str='true', gamedate: str='20230509', gameid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call will pull game scores and no other stats.  
		/getNBAScoresOnly
		Call it with no parameters, it will return the current date's game list with their scores.  
		
		You can use both gameID and gameDate for parameters.  gameID will give you only the scores for one specific game.  gameDate will give you the scores for every game on that date.  
		
		Example, to get all games for March 11, 2022
		/getNBAScoresOnly?gameDate=20220311
		or you can use this call to get just one specific game 
		/getNBAScoresOnly?gameID=20220311_DAL@HOU
		or call with no parameters for the games for 'current processing day'.
		
		If you want to include the top performers from each team when game is in progress, include  topPerformers paremter. 
		 It needs to be &topPerformers=true"
    topperformers: use topPerformers=true to add the game's top performers to each game. If game is in progress or completed, then it will give stats for that game. If game is scheduled then it will give the season average for those players.
        gamedate: format YYYYMMDD
gameDate or gameID is required
Returns all games for the date you ask it for
        gameid: gameDate or gameID is required
        
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAScoresOnly"
    querystring = {}
    if topperformers:
        querystring['topPerformers'] = topperformers
    if gamedate:
        querystring['gameDate'] = gamedate
    if gameid:
        querystring['gameID'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-fantasy-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

