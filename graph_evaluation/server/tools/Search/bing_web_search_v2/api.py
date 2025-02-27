import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def web_search(q: str, responsefilter: str=None, cc: str=None, mkt: str='en-us', promote: str=None, textdecorations: bool=None, setlang: str=None, count: int=None, safesearch: str='Off', textformat: str='Raw', offset: int=None, answercount: int=None, freshness: str='Day', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Bing Web Search API enables you to integrate Bing's search capabilities in your applications. By sending search queries with the API, you can get relevant webpages, images, videos, news and more."
    q: The user's search query term. The term may not be empty. The term may contain Bing Advanced Operators. For example, to limit results to a specific domain, use the site: operator.
        responsefilter: A comma-delimited list of answers to include in the response. If you do not specify this parameter, the response includes all search answers for which there's relevant data. Possible filter values are Computation, Images, News, RelatedSearches, SpellSuggestions, TimeZone, Videos, Webpages. Although you may use this filter to get a single answer, you should instead use the answer-specific endpoint in order to get richer results. For example, to receive only images, send the request to one of the Image Search API endpoints. The RelatedSearches and SpellSuggestions answers do not support a separate endpoint like the Image Search API does (only the Web Search API returns them). To include answers that would otherwise be excluded because of ranking, see the promote query parameter.
        cc: A 2-character country code of the country where the results come from. This API supports only the United States market. If you specify this query parameter, it must be set to us. If you set this parameter, you must also specify the Accept-Language header. Bing uses the first supported language it finds from the languages list, and combine that language with the country code that you specify to determine the market to return results for. If the languages list does not include a supported language, Bing finds the closest language and market that supports the request, or it may use an aggregated or default market for the results instead of a specified one. You should use this query parameter and the Accept-Language query parameter only if you specify multiple languages; otherwise, you should use the mkt and setLang query parameters. This parameter and the mkt query parameter are mutually exclusive—do not specify both.
        mkt: The market where the results come from. Typically, mkt is the country where the user is making the request from. However, it could be a different country if the user is not located in a country where Bing delivers results. The market must be in the form <language code>-<country code>. For example, en-US. The string is case insensitive. If known, you are encouraged to always specify the market. Specifying the market helps Bing route the request and return an appropriate and optimal response. If you specify a market that is not listed in Market Codes, Bing uses a best fit market code based on an internal mapping that is subject to change. This parameter and the cc query parameter are mutually exclusive—do not specify both.
        promote: A comma-delimited list of answers that you want the response to include regardless of their ranking. For example, if you set answerCount) to two (2) so Bing returns the top two ranked answers, but you also want the response to include news, you'd set promote to news. If the top ranked answers are webpages, images, videos, and relatedSearches, the response includes webpages and images because news is not a ranked answer. But if you set promote to video, Bing would promote the video answer into the response and return webpages, images, and videos. The answers that you want to promote do not count against the answerCount limit. For example, if the ranked answers are news, images, and videos, and you set answerCount to 1 and promote to news, the response contains news and images. Or, if the ranked answers are videos, images, and news, the response contains videos and news. Possible values are Computation, Images, News, RelatedSearches, SpellSuggestions, TimeZone, Videos, Webpages. Use only if you specify answerCount.
        textdecorations: A Boolean value that determines whether display strings should contain decoration markers such as hit highlighting characters. If true, the strings may include markers. The default is false. To specify whether to use Unicode characters or HTML tags as the markers, see the textFormat query parameter.
        setlang: The language to use for user interface strings. Specify the language using the ISO 639-1 2-letter language code. For example, the language code for English is EN. The default is EN (English). Although optional, you should always specify the language. Typically, you set setLang to the same language specified by mkt unless the user wants the user interface strings displayed in a different language. This parameter and the Accept-Language header are mutually exclusive; do not specify both. A user interface string is a string that's used as a label in a user interface. There are few user interface strings in the JSON response objects. Also, any links to Bing.com properties in the response objects apply the specified language.
        count: The number of search results to return in the response. The default is 10 and the maximum value is 50. The actual number delivered may be less than requested.Use this parameter along with the offset parameter to page results.For example, if your user interface displays 10 search results per page, set count to 10 and offset to 0 to get the first page of results. For each subsequent page, increment offset by 10 (for example, 0, 10, 20). It is possible for multiple pages to include some overlap in results.
        safesearch: A filter used to filter adult content. Off: Return webpages with adult text, images, or videos. Moderate: Return webpages with adult text, but not adult images or videos. Strict: Do not return webpages with adult text, images, or videos. The default is Moderate. If the request comes from a market that Bing's adult policy requires that safeSearch is set to Strict, Bing ignores the safeSearch value and uses Strict. If you use the site: query operator, there is the chance that the response may contain adult content regardless of what the safeSearch query parameter is set to. Use site: only if you are aware of the content on the site and your scenario supports the possibility of adult content.
        textformat: The type of markers to use for text decorations (see the textDecorations query parameter). Possible values are Raw—Use Unicode characters to mark content that needs special formatting. The Unicode characters are in the range E000 through E019. For example, Bing uses E000 and E001 to mark the beginning and end of query terms for hit highlighting. HTML—Use HTML tags to mark content that needs special formatting. For example, use <b> tags to highlight query terms in display strings. The default is Raw. For display strings that contain escapable HTML characters such as <, >, and &, if textFormat is set to HTML, Bing escapes the characters as appropriate (for example, < is escaped to &lt;).
        offset: The zero-based offset that indicates the number of search results to skip before returning results. The default is 0. The offset should be less than (totalEstimatedMatches - count). Use this parameter along with the count parameter to page results. For example, if your user interface displays 10 search results per page, set count to 10 and offset to 0 to get the first page of results. For each subsequent page, increment offset by 10 (for example, 0, 10, 20). it is possible for multiple pages to include some overlap in results.
        answercount: The number of answers that you want the response to include. The answers that Bing returns are based on ranking. For example, if Bing returns webpages, images, videos, and relatedSearches for a request and you set this parameter to two (2), the response includes webpages and images.If you included the responseFilter query parameter in the same request and set it to webpages and news, the response would include only webpages.
        freshness: Filter search results by the following age values: Day—Return webpages that Bing discovered within the last 24 hours. Week—Return webpages that Bing discovered within the last 7 days. Month—Return webpages that discovered within the last 30 days. This filter applies only to webpage results and not to the other results such as news and images.
        
    """
    url = f"https://bing-web-search1.p.rapidapi.com/search"
    querystring = {'q': q, }
    if responsefilter:
        querystring['responseFilter'] = responsefilter
    if cc:
        querystring['cc'] = cc
    if mkt:
        querystring['mkt'] = mkt
    if promote:
        querystring['promote'] = promote
    if textdecorations:
        querystring['textDecorations'] = textdecorations
    if setlang:
        querystring['setLang'] = setlang
    if count:
        querystring['count'] = count
    if safesearch:
        querystring['safeSearch'] = safesearch
    if textformat:
        querystring['textFormat'] = textformat
    if offset:
        querystring['offset'] = offset
    if answercount:
        querystring['answerCount'] = answercount
    if freshness:
        querystring['freshness'] = freshness
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-web-search1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

