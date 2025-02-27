import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_youtube_video_description(keyword1: str, keyword2: str, include_emojis: str, video_title: str, keyword5: str=None, facebook_link: str='https://www.facebook.com/', linkedin_link: str='https://www.linkedin.com', tiktok_link: str='https://www.tiktok.com/en/', twitter_link: str='https://twitter.com/home?lang=en', instagram_link: str='https://www.instagram.com/', keyword4: str=None, keyword3: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint: /generate_youtube_video_description
		
		Method: GET
		
		Summary: This endpoint generates an SEO-optimized, keyword-rich description for a YouTube video. It also optionally includes social media links.
		
		Parameters:
		
		    video_title (required): The title of the YouTube video.
		    include_emojis (required): A boolean indicating whether to include emojis in the video description.
		    keyword1 to keyword5 (required for keyword1 and keyword2, optional for keyword3 to keyword5): Keywords relevant to the video. A minimum of two keywords and a maximum of five can be specified. Each keyword should not exceed 30 characters.
		    facebook_link, twitter_link, instagram_link, linkedin_link, tiktok_link (all optional): Valid URLs pointing to respective social media profiles.
		
		Response: The response is a YouTubeDescriptionResponse model that includes the video title, video description (which includes the social media links if provided), a boolean indicating if emojis were included, and a list of the provided keywords.
		
		Use Case: Use this endpoint when you need to generate a unique, keyword-rich description for a YouTube video. This can be particularly useful for SEO and improving the visibility of your video in YouTube search results. Also, by providing social media links, you can direct your audience to your other platforms, enhancing your overall online presence.
		
		Errors:
		
		    400 Bad Request: Raised if the number of keywords is not between 2 and 5, or if a keyword exceeds 30 characters.
		    500 Internal Server Error: Raised for unknown API errors.
		    503 Service Unavailable: Raised if the OpenAI server is unavailable."
    
    """
    url = f"https://blogzee-ai-seo-social-content-generator.p.rapidapi.com/generate_youtube_video_description"
    querystring = {'keyword1': keyword1, 'keyword2': keyword2, 'include_emojis': include_emojis, 'video_title': video_title, }
    if keyword5:
        querystring['keyword5'] = keyword5
    if facebook_link:
        querystring['facebook_link'] = facebook_link
    if linkedin_link:
        querystring['linkedin_link'] = linkedin_link
    if tiktok_link:
        querystring['tiktok_link'] = tiktok_link
    if twitter_link:
        querystring['twitter_link'] = twitter_link
    if instagram_link:
        querystring['instagram_link'] = instagram_link
    if keyword4:
        querystring['keyword4'] = keyword4
    if keyword3:
        querystring['keyword3'] = keyword3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blogzee-ai-seo-social-content-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_blog_post(include_html: bool, title: str, keyword1: str, keyword2: str, include_emojis: bool, keyword3: str=None, keyword5: str=None, keyword4: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the /generate_blog_post endpoint for the Blogzee AI service, designed to generate a complete blog post based on user-provided input parameters.
		
		Endpoint Structure:
		
		    Method: GET
		    Route: /generate_blog_post
		    Response Model: BlogPostResponse
		
		Functionality:
		This endpoint accepts a blog post title, a list of keywords (minimum of 2, maximum of 5), and two boolean flags (include_html, include_emojis) as parameters. It uses these inputs to generate a complete blog post using the underlying AI model.
		
		Input Validation:
		The endpoint expects the keywords to be a list with a minimum length of 2 and a maximum length of 5, with each keyword being no longer than 30 characters. If these conditions are not met, an error is thrown.
		
		Error Handling:
		The endpoint has a mechanism to handle potential API errors from OpenAI's language model. If there is a "Request failed due to server shutdown" error, it responds with an HTTP 503 status code and a "Service Unavailable: API server shutdown" message. For other unknown errors, it returns an HTTP 500 status code with a "Internal Server Error: Unknown API error" message.
		
		Output Formatting:
		If the 'include_html' flag is set, it converts new lines in the generated blog post to HTML break tags for better web display.
		
		Output:
		This endpoint returns a BlogPostResponse, which includes the blog post title, the generated blog post content, and the values of include_html and include_emojis flags, and the keywords used for generation.
		
		Overall, this endpoint provides a way for users to generate complete blog posts tailored to specific titles and keywords, significantly accelerating the content creation process with the help of AI."
    keyword1: First keyword, max 30 characters
        keyword2: Second keyword, max 30 characters
        keyword3: Optional third keyword, max 30 characters
        keyword5: Optional fifth keyword, max 30 characters
        keyword4: Optional fourth keyword, max 30 characters
        
    """
    url = f"https://blogzee-ai-seo-social-content-generator.p.rapidapi.com/generate_blog_post"
    querystring = {'include_html': include_html, 'title': title, 'keyword1': keyword1, 'keyword2': keyword2, 'include_emojis': include_emojis, }
    if keyword3:
        querystring['keyword3'] = keyword3
    if keyword5:
        querystring['keyword5'] = keyword5
    if keyword4:
        querystring['keyword4'] = keyword4
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blogzee-ai-seo-social-content-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_blog_post_ideas(keyword1: str, keyword2: str, keyword5: str=None, keyword4: str=None, keyword3: str='student', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the /generate_blog_ideas endpoint for the Blogzee AI service, designed to generate blog post ideas based on user-provided keywords.
		
		Endpoint Structure:
		
		    Method: GET
		    Route: /generate_blog_ideas
		    Response Model: BlogIdeasResponse
		
		Functionality:
		This endpoint accepts a list of keywords (minimum of 2, maximum of 5), each with a maximum length of 30 characters. The keywords are used to generate blog post ideas using the underlying AI model.
		
		Input Validation:
		The endpoint validates the input parameters to ensure they are within the expected range and character limits. If the validation fails, it returns an HTTP 400 status code with an appropriate error message.
		
		Error Handling:
		The endpoint handles potential API errors from OpenAI's language model, specifically the "Request failed due to server shutdown" error. In this case, it returns an HTTP 503 status code with the detail "OpenAI API is currently unavailable". If an unknown error occurs, it returns an HTTP 500 status code.
		
		Output:
		This endpoint returns a BlogIdeasResponse, which is a list of blog post ideas generated based on the input keywords.
		
		Overall, this endpoint provides a way for users to leverage AI to generate creative blog post ideas tailored to specific keywords, helping to streamline the content creation process."
    
    """
    url = f"https://blogzee-ai-seo-social-content-generator.p.rapidapi.com/generate_blog_ideas"
    querystring = {'keyword1': keyword1, 'keyword2': keyword2, }
    if keyword5:
        querystring['keyword5'] = keyword5
    if keyword4:
        querystring['keyword4'] = keyword4
    if keyword3:
        querystring['keyword3'] = keyword3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blogzee-ai-seo-social-content-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

