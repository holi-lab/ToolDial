import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_result_with_input_as_path_parameter(input: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes the initial board configuration as the path parameter input in an encoded format and returns the result as a JSON object. Suppose the board is not initially filled then the input should be a string "x81x" where any number written between two "x" characters is the number of empty cells in the grid. This number should only be placed between two "x" characters only. The number of empty cell can range anywhere from 0 to 81. The input should start from top-leftmost cell and then moving row-wise reading the entire row completely before moving on to the next row and so on. The input given should follows the rules of sudoku and the input constraints given above otherwise it will show invalid input. The expanded string should only be of 81 length.
		If initially the board is ![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg/1200px-Sudoku_Puzzle_by_L2G-20050714_standardized_layout.svg.png) (Photo credit: Wikipedia) then the input should be "53x2x7x4x6x2x195x4x98x4x6x1x8x3x6x3x34x2x8x1x3x2x17x3x2x3x6x1x6x4x28x4x419x2x5x4x8x2x79" in the path parameter then the output JSON answer property will be "534678912672195348198342567859761423426853791713924856961537284287419635345286179"."
    
    """
    url = f"https://sudoku-solver2.p.rapidapi.com/{input}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-solver2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

