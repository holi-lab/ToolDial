import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_customized_plan(fitness_goals: str, fitness_level: str, muscle: str, equipment: str, time: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /customized endpoint provides a way for users to generate personalized workout plans based on their preferences. It takes various parameters into account to create a workout plan tailored to the user's needs. 
		Here's an explanation of the parameters and how to use them:
		1. time: This parameter represents the duration of the workout plan in minutes. By default, if not provided, it is set to 30 minutes. You can customize this parameter to specify the desired duration of your workout.
		2. equipment : This parameter allows you to specify the equipment available for your workout. If you don't have any equipment, you can set it to "none". However, if you have specific equipment available, you can provide the details in this parameter.
		3. muscle: This parameter represents the target muscle group for your workout plan. You can specify the muscle group you want to focus on, such as "legs", "chest", "back", etc. If you don't have a specific target muscle group, you can set it to "none".
		4. fitness_level: This parameter enables you to define your fitness level. You can choose from options like "beginner", "intermediate", or "advanced" to indicate your current fitness level. The workout plan generated will be adjusted based on this level. 
		5. fitness_goals: This parameter allows you to specify your fitness goals, such as "weight_loss", "muscle_gain", "strength_training", "cardiovascular_endurance", "flexibility", or "general_fitness". By indicating your goals, the generated workout plan will align with your specific objectives."
    
    """
    url = f"https://workout-planner1.p.rapidapi.com/customized"
    querystring = {'fitness_goals': fitness_goals, 'fitness_level': fitness_level, 'muscle': muscle, 'equipment': equipment, 'time': time, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "workout-planner1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_workout_plan(time: str, equipment: str, muscle: str, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Workout Plans
		
		This api request can be used to get Workout plan based on:
		
		- Time duration
		- Target Muscle
		- Location
		- Equipment
		
		Hence api call takes 4 query parameters:
		
		- time
		- muscle
		- location
		- equipment"
    
    """
    url = f"https://workout-planner1.p.rapidapi.com/"
    querystring = {'time': time, 'equipment': equipment, 'muscle': muscle, 'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "workout-planner1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

