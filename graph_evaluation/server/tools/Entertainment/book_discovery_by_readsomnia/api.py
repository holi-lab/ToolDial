import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def book_search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A quick search for a book by entering either 1.) the book title or 2.) the author's name."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/search/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def book_data(bid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The most comprehensive book information data collection available through one API Endpoint. It (not only) provides essential information like ISBN10 and ISBN13 bat lots of extra data, e.g.:
		- an informative synopsis, 
		- editorial and user reviews,
		- even video reviews, if exists, 
		- related book suggestions to help users make informed decisions
		and more."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/book/{bid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hachette_global_publisher_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The latest  Hachette Book Bestellers list featuring a curated selection of new and trending bestsellers across a wide range of genres and topics, handpicked by one of the leading publishers in the industry."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_hachette-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harpercollins_global_publisher_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "HarperCollins Bestsellers List showcases the most in-demand books across genres and categories, selected using sales data and expert recommendations from one of the most reputable and recognized booksellers in the industry."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_harper-collins-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_house_global_publisher_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Penguin Random House Bestsellers List, featuring the most popular books of the moment across various genres and categories, curated based on sales data and expert recommendations from one of the most well-known and respected booksellers in the industry."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_random-house-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amazon_bestsellers_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Amazon.com Book Blockbusters List, featuring the most popular books of the moment across various genres and categories, curated based on sales data and expert recommendations from one of the most well-known and respected booksellers in the industry."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_amazon-book-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def barnes_noble_book_blockbusters(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Barnes & Noble Bestsellers List, featuring the most popular books of the moment across various genres and categories, curated based on sales data and expert recommendations from one of the most well-known and respected booksellers in the industry."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_burns-noble-book-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publishers_weekly_bestseller_lists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Publishers Weekly Bestsellers List, featuring the most popular books across various genres and categories, based on sales data and expert recommendations from one of the most respected publishing industry authorities."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_publishers-weekly-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_york_times_best_sellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current List of New York Times Best Sellers Sorted by Genre"
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_new-york-times-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_must_reads_accross_the_board(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### API's Root Endpoint - Ultimate Aggregated Must-Read Collection
		**Instantly get access to the Top Selection of the up-to-date Must-Read books arranged by multiple criterias:**
		1. Top of the Charts Fiction
		2. Top of the Charts Non-Fiction  ↴
		3. Fiction Global Trendsetters
		4. Non-Fiction Global Trendsetters
		5. New York Times Best Sellers
		6. Publishers Weekly Bestseller Lists
		7. Barnes & Noble Book Blockbusters
		8. Amazon.com Book Blockbusters
		9. Top Young Adult Books
		10. Critically Acclaimed Books
		11. Community Acclaimed Must Reads
		12. New & Popular Fiction Bestsellers
		13. New & Popular Non-Fiction Bestsellers
		14. Random House Global Publisher Bestsellers
		15. HarperCollins Global Publisher Bestsellers
		16. Hachette Global Publisher Bestsellers
		17. Independent - INDI Publishers Bestsellers
		18. Children's Bestselling Books
		19. Greatest Books of All Time
		20. Greatest Books of the Millennium"
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/front"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_books_of_the_millennium(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Curated list of must-read fiction and non-fiction books that have had a significant impact in the last century based on recommendations from a variety of experts, authors, and critics in the field."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_millennium-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def children_s_bestselling_books(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "CherryPicked & up-to-date selections of the most popular and beloved children's books from a veriaty of most-reputable sources."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_top-children-books"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def young_adult_best_sellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fresh selection of bestselling fiction books for teens and young adults by the most popular authors that includes hot new series, graphic novels, and new YA releases."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_top-young-adult-books"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def greatest_books_of_all_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated selection of the best fiction and non-fiction books of all time, carefully chosen based on recommendations from various experts, authors, and critics in the field."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_all-time-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def independent_indi_publishers_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "CherryPicked selections of fresh and highly regarded by readers indie fiction and non-fiction bestsellers on sale in hundreds of independent bookstores nationwide."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_indie-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def community_favorites(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fresh selection of books that are highly regarded by readers and have received positive reviews, ratings, and widespread acclaim from the community."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_community-acclaimed-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def award_winning_critically_acclaimed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A fresh collection of the best fiction and non-fiction books, carefully selected based on recommendations from a variety of experts, authors, and critics in the field, using multiple most-reputable sources."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_critically-acclaimed-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_popular_non_fiction_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover incoming Non-Fiction Trendsetters that already dominate the major Global Indi Publishers and media sources' bestseller charts."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_new-non-fiction-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_popular_fiction_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover incoming Fiction Trendsetters that already dominate the major Global Indi Publishers and media sources' bestseller charts."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_new-fiction-bestsellers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_trending_non_fiction_books(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Non-Fiction Trendsetters that currently dominate the major Global Indi Publishers and media sources' bestseller books charts"
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_trending-non-fiction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_trending_fiction_novels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fiction Trendsetters that currently dominate the major Global Indi Publishers and media sources' bestseller books charts"
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_trending-fiction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_of_the_charts_non_fiction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the Top-Ranked Non-Fiction Books that dominate various top-tier Must-Read Charts across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_top-charts-non-fiction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_of_the_charts_fiction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the Best-Ranked Fiction Novels that dominate various top-tier Fiction Leaderboards across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/list_top-charts-fiction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def science_fiction_fantasy_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Science Fiction & Fantasy Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_science-fiction-fantasy"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def biographies_memoirs_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Biographies & Memoirs Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_biographies-memoirs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def self_help_personal_growth_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Self-Help & Personal Growth Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_self-help"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comics_graphic_novels_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Comics & Graphic Novels extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_comics-graphic-novels"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def arts_photography_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Arts & Photography Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_arts-photography"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def humor_entertainment_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Humor & Entertainment Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_humor-entertainment"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def history_chronicles_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best History & Chronicles Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def christian_books_bibles_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Christian Books & Bibles extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_christian-bibles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def children_s_books_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Children's Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_children"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teen_young_adult_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Teen & Young Adult Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_teen-young-adult"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cookbooks_food_wine_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Cookbooks, Food & Wine Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_cookbooks-food-wine"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_money_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Business & Money Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_business-money"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lgbtq_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best LGBTQ+ Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_lgbtq"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health_fitness_dieting_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Health, Fitness & Dieting Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_health-fitness-dieting"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def politics_social_sciences_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Politics & Social Sciences Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_politics-social-science"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def love_romance_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Love & Romance Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_romance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mystery_thriller_suspense_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best Mystery, Thriller & Suspense Books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_mystery-thriller-suspense"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def literature_fiction_bestsellers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A curated & up-to-date list of the best fiction and non-fiction books extrapolated by sophisticated AI algorithms from a variety of top-tier sources across the Internet."
    
    """
    url = f"https://book-discovery-by-readsomnia.p.rapidapi.com/genre_literature-fiction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "book-discovery-by-readsomnia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

