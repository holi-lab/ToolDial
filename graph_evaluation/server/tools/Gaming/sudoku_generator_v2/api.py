import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def solve_a_sudoku_puzzle(puzzle: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint solves a given Sudoku puzzle."
    puzzle: Define the puzzle to solve.

Encode the puzzle as a single line string with 81 chars. Replace blank spaces with a dot.
For example:

`167.83...3...5...8.8....4......2......5...7..6...3...2.2...796.9.6...2...4...2..3`

stands for the puzzle 
`167.83...`
`3...5...8`
`.8....4..`
`....2....`
`..5...7..`
`6...3...2`
`.2...796.`
`9.6...2..`
`.4...2..4`
        
    """
    url = f"https://sudoku-generator1.p.rapidapi.com/sudoku/solve"
    querystring = {'puzzle': puzzle, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_a_sudoku_puzzle(difficulty: str=None, seed: str='1337', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives the following guarantees:
		* The generator is deterministic, so calling the endpoint again with the same parameters, will yield the same puzzle. (except requests without a seed)
		* Each generated puzzle has an unique solution.
		
		See parameters for more information."
    difficulty: The generator has 3 difficulty settings:
* easy (default): the puzzle can be solved with the most basic strategies and there are mutliple ways to solve the puzzle.
* medium:  the puzzle needs more advanced strategies to be solved.
* heard: the puzzle needs advanced strategies to be solved.

The generator does not garantee, that a puzzle falls into this category. It may be easier to solve then defined, but it never is more difficult then defined.
        seed: Define the seed for the randomness used by the generator.

It may be an integer, a string or be omitted. 
        
    """
    url = f"https://sudoku-generator1.p.rapidapi.com/sudoku/generate"
    querystring = {}
    if difficulty:
        querystring['difficulty'] = difficulty
    if seed:
        querystring['seed'] = seed
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-generator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

