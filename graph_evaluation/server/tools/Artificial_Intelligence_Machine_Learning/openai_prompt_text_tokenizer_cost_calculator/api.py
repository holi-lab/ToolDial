import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_openai_models_pricing_information(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves a list of OpenAI models and their associated OpenAI pricing information. You can use the model names in this list when calling the Encode or Decode end-points depending on your needs. You can also use the OpenAI pricing information to do your own calculations.
		
		**Note:**
		- Certain models have extremely big numbers for the **promptCostPerToken **fields, so far it's only applicable to the ADA related models. In this case it's returned as a floating point precision number in scientific notation or exponential notation.
		- Models with 0 cost mean that there is no cost information available on OpenAIs website."
    
    """
    url = f"https://openai-prompt-text-tokenizer-cost-calculator.p.rapidapi.com/openai/chatgpt/models"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "openai-prompt-text-tokenizer-cost-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

