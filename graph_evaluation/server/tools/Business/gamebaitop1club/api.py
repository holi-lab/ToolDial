import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def game_b_i_top_1(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Game Bài Top 1 là một trang chuyên đánh giá, phân tích và xếp hạng các cổng game bài đổi thưởng, nhà cái quốc tế hàng đầu dành cho các anh em có đam mê.
		Mọi thông tin liên quan đến link tải các slots game đổi thưởng, game bắn cá đổi thưởng,... Đặc biệt là những game bài đổi thưởng hấp dẫn, các chương trình khuyến mãi, ưu đãi hấp dẫn, giftcode giá trị.
		Game Bài Top 1 sẽ liên tục cập nhật những thông tin mới nhất dành cho những anh em game thủ có nhu cầu săn code khuyến mãi giá trị, hàng loạt ưu đãi khủng.
		Truy cập ngay Game Bài Top 1 mọi lúc mọi nơi, sẵn sàng trở thành đại gia sau một đêm ngay thôi!
		#gamebaitop1 #Bắn_cá_đổi_thưởng #Nổ_hũ_đổi_thưởng #Nhà_cái_uy_tín #gamebaitop1
		Địa chỉ : Ng. 303 Đường Phúc Lợi, thôn Trung, Phúc Lợi, Long Biên, Hà Nội, Việt Nam
		Google map https://www.google.com/maps?cid=3708580958303477716 
		Hotline 0876943943
		Gmail: gamebaitop1club@gmail.com
		Website https://gamebaitop1.club/ 
		https://gamebaitop1.club/ban-ca-doi-thuong/ 
		https://gamebaitop1.club/no-hu-doi-thuong/ 
		https://gamebaitop1.club/nha-cai-uy-tin/ 
		https://www.facebook.com/gamebaitop1club 
		https://sites.google.com/view/gamebaitop1club 
		https://open.spotify.com/show/2fP5p0AQVcDK0CG1dM551w 
		https://www.tetongravity.com/community/profile/m2fd10/
		https://www.mobypicture.com/user/gamebaitop1club
		https://www.getrevue.co/profile/gamebaitop1club
		https://kyteapp.mn.co/members/11676862
		https://fundrazr.com/profiles/gamebaitop1-club
		https://gamebaitop1club.gitbook.io/welcome-to-gitbook/
		https://www.gaiaonline.com/profiles/gamebaitop1club/45900819/
		https://www.bhimchat.com/gamebaitop1club
		https://pawoo.net/@gamebaitop1club
		https://www.hackathon.io/gamebaitop1club
		https://www.bakespace.com/members/profile/gamebaitop1club/1486664/
		https://guides.co/p/gamebaitop1-club
		https://www.allmyfaves.com/gamebaitop1club/
		https://my.mamul.am/en/profile/3101300/info
		https://kuwestions.248am.com/user/gamebaitop1club
		https://www.mapleprimes.com/users/gamebaitop1club
		https://www.bahamaslocal.com/userprofile/1/135524/gamebaitop1club.html
		https://www.misterpoll.com/forums/1/topics/331373
		https://yarabook.com/gamebaitop1club
		https://anchor.fm/gamebaitop1club
		https://www.bitsdujour.com/profiles/etPBhc
		https://www.techrum.vn/members/gamebaitop1club.204521/
		https://gamebaitop1club.netboard.me/
		https://socialcompare.com/en/member/gamebaitop1club-6i4egngj
		https://git.qoto.org/gamebaitop1club
		https://www.gapo.vn/197684172
		https://www.bbuzzart.com/profile/436843
		https://faithlife.com/gamebaitop1club
		http://kiredu.ru/UserProfile/tabid/182/userId/88108/Default.aspx
		https://www.thebranfordgroup.com/dnn3/UserProfile/tabid/214/UserId/62555/Default.aspx
		https://beermapping.com/account/gamebaitop1club
		http://gendou.com/user/gamebaitop1club
		https://sumally.com/gamebaitop1club
		https://www.furaffinity.net/user/gamebaitop1club/
		https://www.bitrated.com/gamebaitop1club
		http://yourlisten.com/gamebaitop1club
		https://www.plasterersforum.com/members/gamebaitop1club.78337/"
    
    """
    url = f"https://gamebaitop1club.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamebaitop1club.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

