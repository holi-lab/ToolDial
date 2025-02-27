import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_daily_scoreboard_live_real_time(topperformers: str='true', gamedate: str='20230410', gameid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this if you want basic game data returned.  It's lighter/quicker than getting the full boxscore, for applications that do not need anything but basic data like line score, away/home, etc.
		
		/getMLBScoresOnly
		
		This can be called using ?gameDate (returns all games for a date, format YYYYMMDD)  or ?gameID (returns one game, format YYYYMMDD_AWAY@HOME)
		
		Also can be called with topPerformers=true to get a list of stat leaders in each category, per team."
    topperformers: true
If the game has started, then this will give you the top performers for the game. If not, then it gives top performers from each team based on season stats.
        
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBScoresOnly"
    querystring = {}
    if topperformers:
        querystring['topPerformers'] = topperformers
    if gamedate:
        querystring['gameDate'] = gamedate
    if gameid:
        querystring['gameID'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_news_and_headlines(maxitems: str='10', fantasynews: str=None, recentnews: str='true', topnews: str=None, playerid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve relevant news links.
		/getMLBNews
		
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
    maxitems: number
        fantasynews: true
        recentnews: true
        topnews: true
        playerid: playerID as can be found in many of the endpoints
        
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBNews"
    querystring = {}
    if maxitems:
        querystring['maxItems'] = maxitems
    if fantasynews:
        querystring['fantasyNews'] = fantasynews
    if recentnews:
        querystring['recentNews'] = recentnews
    if topnews:
        querystring['topNews'] = topnews
    if playerid:
        querystring['playerID'] = playerid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_player_information(getstats: str='false', playerid: str=None, playername: str='abreu', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call this to get general information on each player (name, team, experience, birthday, college, image link, etc).
		
		This can accept either "playerID" or "playerName".    
		
		If you use playerID then the body will return one object.  playerID values can be found from performing a get on the team roster API. playerID is the unique identifier for each player, and is the preferred parameter to use in this call.
		
		If you use playerName then it will return a list of objects, since many players can have the same name.  It acts as more of a search/scan than direct access, and will be a slower call than if you use playerID.
		
		Also, you don't have to call the full name with playerName.  You can use partial name.  For example, if you call with only playerName=smith then it will return all players with smith in their full name.
		
		getStats=true will bring back current season stats for the returned players"
    getstats: getStats=true will bring back each current season stats for the returned players
        
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBPlayerInfo"
    querystring = {}
    if getstats:
        querystring['getStats'] = getstats
    if playerid:
        querystring['playerID'] = playerid
    if playername:
        querystring['playerName'] = playername
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_roster(getstats: str='true', teamabv: str='CHW', teamid: str=None, archivedate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns the current or historical* roster of any team, using the teamID that can be found in "getMLBTeams" call.
		
		Rosters are updated hourly during the day.
		
		Historical rosters are saved on a daily basis as of 20230505 and moving forward. 
		 
		Here are examples of the two ways to call and get the White Sox roster:
		/getMLBTeamRoster?teamID=6
		or
		/getMLBTeamRoster?teamAbv=CHW
		
		add getStats=true to get player stats.  
		
		Add parameter archiveDate to the call to get a list of roster players (playerID's only) for that specific date.  Historical roster dates only are kept as far back as 20230505.
		
		getStats does not work with the historical Roster"
    getstats: getStats=true for player stats. Does not work with historical roster call.
        
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBTeamRoster"
    querystring = {}
    if getstats:
        querystring['getStats'] = getstats
    if teamabv:
        querystring['teamAbv'] = teamabv
    if teamid:
        querystring['teamID'] = teamid
    if archivedate:
        querystring['archiveDate'] = archivedate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams(topperformers: str='true', teamstats: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides you with all teams, their cities, names, abbreviations, and some other general information.  This endpoint isn't necessary to call very often, as the data here won't change much throughout the season.  Perhaps make a point to call it once a week, as we will most likely be adding information to it every once in a while.
		
		This does not need parameters. It just returns the same full list every time. 
		
		Optional parameters are... 
		
		topPerformers=true 
		This will return the best player for each statistical category on each team.
		
		teamStats=true
		
		This will return team level/season long stats for each team"
    topperformers: topPerformers=true to retrieve the best player in each category for each team
        teamstats: teamStats=true to retrieve Team Level/season long statistics for each team.

        
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBTeams"
    querystring = {}
    if topperformers:
        querystring['topPerformers'] = topperformers
    if teamstats:
        querystring['teamStats'] = teamstats
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mlb_betting_odds(playerprops: str='true', gamedate: str='20230703', gameid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This grabs MLB betting/gambling lines and odds from some of the most popular online sportsbooks (fanduel, betrivers, betmgm, caesars, pointsbet, etc). 
		
		Player prop bets can be added by including playerProps=true parameter.  Prop bets are new as of 7/4/2023.  Happy 4th! 
		
		You can call this for specific game or a specific date.  Check out the example responses here for the type of data you can expect back.   Some of the sportsbooks do not offer live betting, so data from those sportsbooks will not be returned after the game starts.  
		
		
		Either gameDate or gameID is required.
		Examples of what the calls can look like:
		/getMLBBettingLines?gameDate=20230410
		/getMLBBettingLines?gameID=20230410_HOU@PIT"
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBBettingOdds"
    querystring = {}
    if playerprops:
        querystring['playerProps'] = playerprops
    if gamedate:
        querystring['gameDate'] = gamedate
    if gameid:
        querystring['gameID'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_player_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "One call grabs the FULL MLB player list. This is mainly used for associating players with their "playerID" which is what you'll want to use when cross referencing with box scores.
		
		No parameters, just make the call:
		
		/getMLBPlayerLIst"
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBPlayerList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_mlb_games_and_stats_for_a_single_player(season: str='2023', numberofgames: str=None, gameid: str=None, playerid: str='592450', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call will grab a map of all of the games a player has played this season.   
		playerID is a required parameter.
		You can also use gameID if you want to only pull back a specific game. 
		
		season is an optional parameter.  Currently only 2022 (last season) and 2023 (this season) are available.  If you do not include season as a parameter, it will return this season's games. 
		
		
		You can limit the amount of games returned with parameter: numberOfGames.   For example: &numberOfGames=5 will return the last 5 games this player has an entry for.
		
		Example:
		Correct way to get the stats for Aaron Judge for the season opener against SF on 3/30/2023, would be this:
		/getMLBGamesForPlayer?playerID=592450&gameID=20230330_SF@NYY
		
		But if you wanted to get all of his games this season, you'd make this call
		/getMLBGamesForPlayer?playerID=592450
		
		This call will not work without playerID.  If you want stats for all players during a game, then use the getMLBBoxScore call with that specific gameID."
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBGamesForPlayer"
    querystring = {}
    if season:
        querystring['season'] = season
    if numberofgames:
        querystring['numberOfGames'] = numberofgames
    if gameid:
        querystring['gameID'] = gameid
    if playerid:
        querystring['playerID'] = playerid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_schedule(season: str='2023', teamid: str=None, teamabv: str='CHW', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the full season schedule for any MLB team identified in the parameters.
		
		Example:
		
		/getMLBTeamSchedule?teamID=6
		or
		/getMLBTeamSchedule?teamAbv=CHW
		
		Calling it either way will return the same result, a list of the White Sox games this season, each game in it's own map.  If the game has been played, the linescore and game result will be included in the game's map.
		
		You can also add the "season" parameter if you want to specify season.  Right now we only have season 2022 and 2023.   
		Default season is current season."
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBTeamSchedule"
    querystring = {}
    if season:
        querystring['season'] = season
    if teamid:
        querystring['teamID'] = teamid
    if teamabv:
        querystring['teamAbv'] = teamabv
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_box_score_live_real_time(gameid: str='20230410_CHW@MIN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the entire box score for a game either in progress or already completed.  The stats retrieved here are what are normally shown in box scores or used in fantasy games.  If there are any stats not here that you'd like to see, we can add them.  
		
		The call looks like this /getMLBBoxScore?gameID=20220409_CHW@DET
		
		The call needs to be exactly in the same format as above.  8 digit date, underscore, then the away team abbreviation, @, then home team abbreviation.  Complete list of team abbreviations can be retrieved with the getMLBTeams call or various other calls.  
		
		But, the best way to find specific game ID's are either from the "getMLBGamesForDate" call, or the "getMLBTeamSchedule" call.   
		
		The response will look like this:
		
		`{
		"statusCode":200
		"body":"{"GameLength": "3:11", "Umpires": "HP: Adrian Johnson. 1B: John Tumpane. 2B: Ryan Blakney. 3B: Marvin Hudson.", "playerStats": {"28418863327": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "2", "AB": "2", "battingOrder": "8", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".000", "OPS": ".250", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "2", "LOB": "3"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "1B", "allPositionsPlayed": "1B", "mlbName": "Spencer Torkelson", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28418863327", "longName": "Spencer Torkelson"}, "28738341769": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "3", "battingOrder": "9", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".167", "OPS": ".619", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "2", "LOB": "1"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "3B", "allPositionsPlayed": "3B", "mlbName": "Jake Burger", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28738341769", "longName": "Jake Burger"}, "28808175329": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "", "Flyouts": "1", "Inherited Runners": "0", "H": "0", "HR": "0", "ER": "0", "Strikes": "6", "Inherited Runners Scored": "0", "Groundouts": "0", "R": "0", "ERA": "0.00", "InningsPitched": "1.0", "Batters Faced": "3", "SO": "1", "Pitches": "9"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "DET", "allPositionsPlayed": "P", "mlbName": "Joe Jim\u00e9nez", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28808175329", "longName": "Joe Jimenez"}, "28986584052": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "2", "battingOrder": "1", "H": "1", "IBB": "0", "HR": "0", "TB": "1", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".571", "OPS": "1.285", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "0", "LOB": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "RF", "allPositionsPlayed": "RF", "mlbName": "AJ Pollock", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28986584052", "longName": "A.J. Pollock"}, "28918608009": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "3", "battingOrder": "9", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "GIDP": "1", "3B": "0", "2B": "0", "R": "0", "AVG": ".000", "OPS": ".000", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "2", "LOB": "3"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "CF", "allPositionsPlayed": "CF", "mlbName": "Akil Baddoo", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28918608009", "longName": "Akil Baddoo"}, "28058605227": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "note": "Ran for Pollock in the 3rd.", "Hitting": {"BB": "0", "AB": "2", "battingOrder": "1", "H": "1", "IBB": "0", "HR": "0", "TB": "1", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".333", "OPS": "1.166", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "1", "SO": "0", "LOB": "1"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "CHW", "allPositionsPlayed": "PR-RF", "mlbName": "Andrew Vaughn", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28058605227", "longName": "Andrew Vaughn"}, "28248934929": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "2", "AB": "2", "battingOrder": "7", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "GIDP": "1", "3B": "0", "2B": "0", "R": "0", "AVG": ".250", "OPS": "1.500", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "1", "LOB": "4"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "C", "allPositionsPlayed": "C", "mlbName": "Eric Haase", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28248934929", "longName": "Eric Haase"}, "28158563969": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "2", "H": "2", "IBB": "0", "HR": "0", "TB": "2", "3B": "0", "GIDP": "0", "2B": "0", "R": "1", "AVG": ".500", "OPS": "1.125", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "0", "LOB": "2"}, "BaseRunning": {"CS": "0", "SB": "1", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "CF", "allPositionsPlayed": "CF", "mlbName": "Luis Robert", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28158563969", "longName": "Luis Robert"}, "28968190527": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "", "Flyouts": "0", "Inherited Runners": "0", "H": "3", "HR": "0", "ER": "1", "Strikes": "12", "Inherited Runners Scored": "0", "Groundouts": "2", "R": "1", "ERA": "13.50", "InningsPitched": "0.2", "Batters Faced": "5", "SO": "0", "Pitches": "21"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "DET", "allPositionsPlayed": "P", "mlbName": "Jason Foley", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28968190527", "longName": "Jason Foley"}, "28138355329": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "5", "H": "1", "IBB": "0", "HR": "0", "TB": "1", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".250", "OPS": ".500", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "2", "SO": "0", "LOB": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "LF", "allPositionsPlayed": "LF", "mlbName": "Eloy Jim\u00e9nez", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28138355329", "longName": "Eloy Jimenez"}, "28978081329": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "2", "Balk": "0", "Wild Pitch": "0", "decision": "", "Flyouts": "0", "Inherited Runners": "1", "H": "2", "HR": "0", "ER": "1", "Strikes": "15", "Inherited Runners Scored": "1", "Groundouts": "2", "R": "1", "ERA": "9.00", "InningsPitched": "1.0", "Batters Faced": "6", "SO": "0", "Pitches": "28"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "CHW", "allPositionsPlayed": "P", "mlbName": "Reynaldo L\u00f3pez", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28978081329", "longName": "Reynaldo Lopez"}, "28618589089": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "L", "Flyouts": "7", "Inherited Runners": "0", "H": "7", "HR": "1", "ER": "4", "Strikes": "51", "Inherited Runners Scored": "0", "Groundouts": "4", "R": "4", "ERA": "7.20", "InningsPitched": "5.0", "Batters Faced": "22", "SO": "2", "Pitches": "81"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "allPositionsPlayed": "P", "mlbName": "Casey Mize", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28618589089", "longName": "Casey Mize"}, "28018056809": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "H", "Flyouts": "0", "Inherited Runners": "0", "H": "1", "HR": "0", "ER": "0", "Strikes": "4", "Inherited Runners Scored": "0", "Groundouts": "1", "R": "0", "ERA": "0.00", "InningsPitched": "0.2", "Batters Faced": "2", "SO": "0", "Pitches": "6"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "CHW", "allPositionsPlayed": "P", "mlbName": "Jos\u00e9 Ruiz", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28018056809", "longName": "Jose Ruiz"}, "28118792149": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "4", "H": "1", "IBB": "0", "HR": "0", "TB": "2", "GIDP": "1", "3B": "0", "2B": "1", "R": "1", "AVG": ".250", "OPS": ".625", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "1", "SO": "1", "LOB": "2"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "3B", "allPositionsPlayed": "3B", "mlbName": "Jeimer Candelario", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28118792149", "longName": "Jeimer Candelario"}, "28546661352": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "4", "H": "1", "IBB": "0", "HR": "1", "TB": "4", "3B": "0", "GIDP": "0", "2B": "0", "R": "1", "AVG": ".143", "OPS": ".821", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "2", "SO": "1", "LOB": "5"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "C", "allPositionsPlayed": "C", "mlbName": "Yasmani Grandal", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28546661352", "longName": "Yasmani Grandal"}, "28536595229": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "note": "Struck out for Baddoo in the 9th.", "Hitting": {"BB": "0", "AB": "1", "battingOrder": "0", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".000", "OPS": ".000", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "1", "LOB": "2"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "DET", "allPositionsPlayed": "PH", "mlbName": "Dustin Garneau", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28536595229", "longName": "Dustin Garneau"}, "28036573429": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "3", "H": "2", "IBB": "0", "HR": "0", "TB": "3", "3B": "0", "GIDP": "0", "2B": "1", "R": "2", "AVG": ".286", "OPS": ".804", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "0", "LOB": "3"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "1B", "allPositionsPlayed": "1B", "mlbName": "Jos\u00e9 Abreu", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28036573429", "longName": "Jose Abreu"}, "28488244062": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "6", "H": "1", "IBB": "0", "HR": "0", "TB": "1", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".286", "OPS": ".804", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "1", "SO": "1", "LOB": "1"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "2B", "allPositionsPlayed": "2B", "mlbName": "Jonathan Schoop", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28488244062", "longName": "Jonathan Schoop"}, "28468725739": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "S", "Flyouts": "0", "Inherited Runners": "0", "H": "2", "HR": "0", "ER": "0", "Strikes": "15", "Inherited Runners Scored": "0", "Groundouts": "0", "R": "0", "ERA": "10.80", "InningsPitched": "1.0", "Batters Faced": "5", "SO": "3", "Pitches": "24"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "CHW", "allPositionsPlayed": "P", "mlbName": "Aaron Bummer", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28468725739", "longName": "Aaron Bummer"}, "28548315769": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "6", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".000", "OPS": ".000", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "0", "LOB": "1"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "DH", "allPositionsPlayed": "DH", "mlbName": "Gavin Sheets", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28548315769", "longName": "Gavin Sheets"}, "28208115769": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "H", "Flyouts": "1", "Inherited Runners": "1", "H": "0", "HR": "0", "ER": "0", "Strikes": "9", "Inherited Runners Scored": "0", "Groundouts": "1", "R": "0", "ERA": "0.00", "InningsPitched": "1.1", "Batters Faced": "3", "SO": "1", "Pitches": "13"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "CHW", "allPositionsPlayed": "P", "mlbName": "Bennett Sousa", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28208115769", "longName": "Bennett Sousa"}, "28808116349": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "2", "H": "2", "IBB": "0", "HR": "0", "TB": "2", "3B": "0", "GIDP": "0", "2B": "0", "R": "1", "AVG": ".500", "OPS": "1.500", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "1", "LOB": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "LF", "allPositionsPlayed": "LF", "mlbName": "Austin Meadows", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28808116349", "longName": "Austin Meadows"}, "28748745449": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "", "Flyouts": "0", "Inherited Runners": "0", "H": "0", "HR": "0", "ER": "0", "Strikes": "7", "Inherited Runners Scored": "0", "Groundouts": "1", "R": "0", "ERA": "0.00", "InningsPitched": "1.0", "Batters Faced": "3", "SO": "2", "Pitches": "11"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "DET", "allPositionsPlayed": "P", "mlbName": "Michael Fulmer", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28748745449", "longName": "Michael Fulmer"}, "28086504532": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "8", "H": "2", "IBB": "0", "HR": "0", "TB": "5", "3B": "1", "GIDP": "0", "2B": "1", "R": "1", "AVG": ".250", "OPS": ".875", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "0", "LOB": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "2B", "allPositionsPlayed": "2B", "mlbName": "Josh Harrison", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28086504532", "longName": "Josh Harrison"}, "28018152299": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "3", "Balk": "0", "Wild Pitch": "0", "decision": "W", "Flyouts": "2", "Inherited Runners": "0", "H": "2", "HR": "0", "ER": "1", "Strikes": "46", "Inherited Runners Scored": "0", "Groundouts": "3", "R": "1", "ERA": "1.80", "InningsPitched": "5.0", "Batters Faced": "20", "SO": "8", "Pitches": "79"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "allPositionsPlayed": "P", "mlbName": "Dylan Cease", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28018152299", "longName": "Dylan Cease"}, "28378135769": {"gameID": "20220409_CHW@DET", "Pitching": {"BB": "0", "Balk": "0", "Wild Pitch": "0", "decision": "", "Flyouts": "0", "Inherited Runners": "2", "H": "0", "HR": "0", "ER": "0", "Strikes": "7", "Inherited Runners Scored": "0", "Groundouts": "3", "R": "0", "ERA": "0.00", "InningsPitched": "1.1", "Batters Faced": "4", "SO": "0", "Pitches": "11"}, "Hitting": {"2B": "0", "SF": "0", "SAC": "0", "HBP": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "DET", "allPositionsPlayed": "P", "mlbName": "Will Vest", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28378135769", "longName": "Will Vest"}, "28808238329": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "note": "Ran for Vaughn in the 7th.", "Hitting": {"BB": "0", "AB": "1", "battingOrder": "1", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".000", "OPS": ".000", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "1", "LOB": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "False", "team": "CHW", "allPositionsPlayed": "PR-RF", "mlbName": "Adam Engel", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28808238329", "longName": "Adam Engel"}, "28178284052": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "7", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".000", "OPS": ".000", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "1", "LOB": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "CHW", "startingPosition": "SS", "allPositionsPlayed": "SS", "mlbName": "Leury Garc\u00eda", "Fielding": {"Passed Ball": "0", "E": "1", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28178284052", "longName": "Leury Garcia"}, "28698930862": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "3", "H": "2", "IBB": "0", "HR": "0", "TB": "2", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".444", "OPS": ".888", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "1", "LOB": "1"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "SS", "allPositionsPlayed": "SS", "mlbName": "Javier B\u00e1ez", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28698930862", "longName": "Javier Baez"}, "28416884532": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "0", "AB": "4", "battingOrder": "1", "H": "0", "IBB": "0", "HR": "0", "TB": "0", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".143", "OPS": ".476", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "0", "LOB": "2"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "RF", "allPositionsPlayed": "RF", "mlbName": "Robbie Grossman", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "28416884532", "longName": "Robbie Grossman"}, "2858676669": {"gameID": "20220409_CHW@DET", "Pitching": {"Balk": "0", "Groundouts": "0", "Wild Pitch": "0", "Flyouts": "0", "Inherited Runners": "0", "Batters Faced": "0", "Pitches": "0", "Strikes": "0", "Inherited Runners Scored": "0"}, "Hitting": {"BB": "1", "AB": "3", "battingOrder": "5", "H": "1", "IBB": "0", "HR": "0", "TB": "1", "3B": "0", "GIDP": "0", "2B": "0", "R": "0", "AVG": ".286", "OPS": ".661", "SF": "0", "SAC": "0", "HBP": "0", "RBI": "0", "SO": "2", "LOB": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "started": "True", "team": "DET", "startingPosition": "DH", "allPositionsPlayed": "DH", "mlbName": "Miguel Cabrera", "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "playerID": "2858676669", "longName": "Miguel Cabrera"}}, "decisions": [{"name": "Dylan Cease", "team": "CHW", "decision": "W", "playerID": "28018152299"}, {"name": "Bennett Sousa", "team": "CHW", "decision": "HLD", "playerID": "28208115769"}, {"name": "Jose Ruiz", "team": "CHW", "decision": "HLD", "playerID": "28018056809"}, {"name": "Aaron Bummer", "team": "CHW", "decision": "S", "playerID": "28468725739"}, {"name": "Casey Mize", "team": "DET", "decision": "L", "playerID": "28618589089"}], "gameStatus": "Completed", "Attendance": "17,469", "teamStats": {"away": {"Pitching": {"Groundouts": "7", "Balk": "0", "Wild Pitch": "0", "Flyouts": "3", "Inherited Runners": "2", "Batters Faced": "36", "Pitches": "150", "Strikes": "89", "Inherited Runners Scored": "1"}, "BaseRunning": {"CS": "0", "SB": "1", "PO": "0"}, "Fielding": {"Passed Ball": "0", "E": "1", "Outfield Assists": "0", "Pickoffs": "0"}, "Hitting": {"BB": "0", "2B": "2", "R": "5", "SF": "0", "SAC": "0", "HBP": "0", "H": "10", "RBI": "5", "IBB": "0", "TB": "17", "3B": "1", "GIDP": "0"}}, "home": {"Pitching": {"Groundouts": "10", "Balk": "0", "Wild Pitch": "0", "Flyouts": "8", "Inherited Runners": "2", "Batters Faced": "37", "Pitches": "133", "Strikes": "83", "Inherited Runners Scored": "0"}, "BaseRunning": {"CS": "0", "SB": "0", "PO": "0"}, "Fielding": {"Passed Ball": "0", "E": "0", "Outfield Assists": "0", "Pickoffs": "0"}, "Hitting": {"BB": "5", "2B": "1", "R": "2", "SF": "0", "SAC": "0", "HBP": "0", "H": "7", "RBI": "2", "IBB": "0", "TB": "8", "3B": "0", "GIDP": "3"}}}, "gameDate": "20220409", "Venue": "Comerica Park", "awayResult": "W", "currentCount": "", "homeResult": "L", "away": "CHW", "Weather": "39 degrees, Cloudy", "lineScore": {"away": {"H": "10", "R": "5", "team": "CHW", "scoresByInning": {"1": "2", "2": "0", "3": "0", "4": "0", "5": "0", "6": "2", "7": "1", "8": "0", "9": "0"}, "E": "1"}, "home": {"H": "7", "R": "2", "team": "DET", "scoresByInning": {"1": "0", "2": "0", "3": "0", "4": "0", "5": "0", "6": "2", "7": "0", "8": "0", "9": "0"}, "E": "0"}}, "currentOuts": "", "FirstPitch": "1:11 PM", "currentInning": "", "gameID": "20220409_CHW@DET", "Wind": "5 mph, Out To CF", "home": "DET"}"
		}`"
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBBoxScore"
    querystring = {}
    if gameid:
        querystring['gameID'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_general_game_information(gameid: str='20230410_CHW@MIN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns very basic information about each game.  The some data points that you will get from this call which you won't get from other calls are the time that the game starts, the game ID and link  for mlb.com and espn.com, and the game status (Postponed/scheduled/completed/in-progress/etc).  All game start times are in Eastern time zone.  
		
		Example:
		
		/getMLBGameInfo?gameID=20220409_CHW@DET
		
		will return:
		
		```
		
		{
		"statusCode":200
		"body":"{"espnID": "401354266", "mlbLink": "https://www.mlb.com/gameday/white-sox-vs-tigers/2022/04/09/662864#game_tab=box,game=662864", "gameStatus": "Completed", "season": "2022", "gameDate": "20220409", "gameTime": "1:10p", "away": "CHW", "mlbID": "662864", "gameID": "20220409_CHW@DET", "espnLink": "https://www.espn.com/mlb/boxscore/_/gameId/401354266", "home": "DET"}"
		}
		```"
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBGameInfo"
    querystring = {}
    if gameid:
        querystring['gameID'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_line_score_real_time(gameid: str='20230410_CHW@MIN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the basic "line score" for a game, whether completed earlier this season, or currently happening now, in real time. 
		
		A baseball line score consists of the basic R/H/E, plus the scores by inning and any pitching scoring decisions. 
		
		Example:
		
		/getMLBLineScore?gameID=20220409_CHW@DET
		
		..will return this:
		
		```
		{
		"statusCode":200
		"body":"{"decisions": [{"name": "Dylan Cease", "team": "CHW", "decision": "W", "playerID": "28018152299"}, {"name": "Bennett Sousa", "team": "CHW", "decision": "HLD", "playerID": "28208115769"}, {"name": "Jose Ruiz", "team": "CHW", "decision": "HLD", "playerID": "28018056809"}, {"name": "Aaron Bummer", "team": "CHW", "decision": "S", "playerID": "28468725739"}, {"name": "Casey Mize", "team": "DET", "decision": "L", "playerID": "28618589089"}], "awayResult": "W", "gameStatus": "Completed", "homeResult": "L", "away": "CHW", "lineScore": {"away": {"H": "10", "R": "5", "team": "CHW", "scoresByInning": {"1": "2", "2": "0", "3": "0", "4": "0", "5": "0", "6": "2", "7": "1", "8": "0", "9": "0"}, "E": "1"}, "home": {"H": "7", "R": "2", "team": "DET", "scoresByInning": {"1": "0", "2": "0", "3": "0", "4": "0", "5": "0", "6": "2", "7": "0", "8": "0", "9": "0"}, "E": "0"}}, "gameDate": "20220409", "gameID": "20220409_CHW@DET", "home": "DET"}"
		}
		```"
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBLineScore"
    querystring = {}
    if gameid:
        querystring['gameID'] = gameid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_daily_schedule(gamedate: str='20230510', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get basic information on which games are being played during a day.
		
		Example call: 
		
		/getMLBGamesForDate?gameDate=20220410
		
		The above call will return all of the games from April 10th, 2022.  Date must be in the format YYYYMMDD.
		
		Games are returned in a list like shown below:
		
		```
		{"statusCode":200
		"body":"[{"gameID": "20220410_BAL@TB", "away": "BAL", "gameDate": "20220410", "home": "TB"}, {"gameID": "20220410_TEX@TOR", "away": "TEX", "gameDate": "20220410", "home": "TOR"}, {"gameID": "20220410_SD@ARI", "away": "SD", "gameDate": "20220410", "home": "ARI"}, {"gameID": "20220410_CHW@DET", "away": "CHW", "gameDate": "20220410", "home": "DET"}, {"gameID": "20220410_HOU@LAA", "away": "HOU", "gameDate": "20220410", "home": "LAA"}, {"gameID": "20220410_PIT@STL", "away": "PIT", "gameDate": "20220410", "home": "STL"}, {"gameID": "20220410_BOS@NYY", "away": "BOS", "gameDate": "20220410", "home": "NYY"}, {"gameID": "20220410_NYM@WAS", "away": "NYM", "gameDate": "20220410", "home": "WAS"}, {"gameID": "20220410_MIL@CHC", "away": "MIL", "gameDate": "20220410", "home": "CHC"}, {"gameID": "20220410_LAD@COL", "away": "LAD", "gameDate": "20220410", "home": "COL"}, {"gameID": "20220410_SEA@MIN", "away": "SEA", "gameDate": "20220410", "home": "MIN"}, {"gameID": "20220410_CLE@KC", "away": "CLE", "gameDate": "20220410", "home": "KC"}, {"gameID": "20220410_OAK@PHI", "away": "OAK", "gameDate": "20220410", "home": "PHI"}, {"gameID": "20220410_MIA@SF", "away": "MIA", "gameDate": "20220410", "home": "SF"}, {"gameID": "20220410_CIN@ATL", "away": "CIN", "gameDate": "20220410", "home": "ATL"}]"
		}
		```"
    
    """
    url = f"https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBGamesForDate"
    querystring = {}
    if gamedate:
        querystring['gameDate'] = gamedate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

