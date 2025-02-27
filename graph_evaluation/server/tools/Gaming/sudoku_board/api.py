import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def solveboard(sudo: str, stype: str='list', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This would solve the inputted sudoku board and return the solution either as a string or as a list (default). The sudoku board to be solved would have to be inputted as a string of characters starting from the first character in the first row and ending in the last character in the last row."
    
    """
    url = f"https://sudoku-board.p.rapidapi.com/solve-board"
    querystring = {'sudo': sudo, }
    if stype:
        querystring['stype'] = stype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-board.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verifyboard(sudo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This would verify if the inputted sudoku board is solvable, the board would have to be inputted as a string of 81 characters, the numbers should start from the first number on the first row of the sudoku board and go sequentially row by row till the last number on the last row of the board which would be the 81st number. The number zero can either be represented as a 0 or a dot (.) or any other non numeric character would still count as a 0."
    
    """
    url = f"https://sudoku-board.p.rapidapi.com/verify-board"
    querystring = {'sudo': sudo, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-board.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generateboard(diff: str='2', solu: str='true', stype: str='list', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This would generate a new sudoku board and provide the output as either a string or a list, if it is provided as a list the list would contain 9 sublists which would be a row in the sudoku board (this is the default). The zero would signify empty squares in the board. But if it is provided as a string then dots (.) would indicate empty squares. Also the solution can be provided if indicated by the user. The difficulty of the board can be specified with 1 being for the easiest board and 3 being for the hardest board, if the parameter put for the difficulty of the board does not fall between the numbers 1 and 3 then an easy board would be returned."
    
    """
    url = f"https://sudoku-board.p.rapidapi.com/new-board"
    querystring = {}
    if diff:
        querystring['diff'] = diff
    if solu:
        querystring['solu'] = solu
    if stype:
        querystring['stype'] = stype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-board.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

