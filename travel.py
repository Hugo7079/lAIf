import streamlit as st
import pandas as pd
import hashlib
import sqlite3
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
import requests
import re
from transformers import pipeline
from snownlp import SnowNLP
import pydeck as pdk

# 設定頁面配置
st.set_page_config(page_title="綜合評論平台", layout="wide")

# 創建資料庫連接
def get_connection():
    conn = sqlite3.connect('database.db', check_same_thread=False)
    return conn

# 創建資料表
def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    # Users 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            credit_score REAL DEFAULT 0.0  -- 初始信用分數為0
        )
    ''')
    # Reviews 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            place_id INTEGER NOT NULL,
            user_id INTEGER,
            review_text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            display INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    # Favorites 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            favorite_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            place_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    # Places 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS places (
            place_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT DEFAULT '地址不詳',
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            category TEXT NOT NULL  -- 新增類別字段，值可以是'restaurant', 'attraction', 'hotel'
        )
    ''')
    conn.commit()
    conn.close()

create_tables()

# 用戶註冊
def register_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # 用戶名已存在
    finally:
        conn.close()

# 用戶登入
def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT user_id FROM users WHERE username = ? AND password = ?', (username, hashed_password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return user[0]  # 返回 user_id
    else:
        return None

# 使用 geopy 獲取城市座標
@st.cache_data
def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="review_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"找不到 {city_name} 的座標")
            return None, None
    except GeocoderServiceError:
        st.error("地理編碼服務出錯，請稍後再試")
        return None, None

# 使用 Overpass API 獲取地點數據（根據城市名稱和類別）
def load_place_data_by_city_and_category(city_name, category):
    try:
        # 獲取城市經緯度和 OSM 資訊
        geolocator = Nominatim(user_agent="review_app")
        location = geolocator.geocode(city_name)
        if not location:
            st.error(f"無法找到城市：{city_name}")
            return pd.DataFrame()

        # 獲取城市的 OSM ID
        osm_id = int(location.raw.get('osm_id'))
        osm_type = location.raw.get('osm_type')
        if osm_type == 'relation':
            area_id = 3600000000 + osm_id
        elif osm_type == 'way':
            area_id = 2400000000 + osm_id
        elif osm_type == 'node':
            area_id = 3600000000 + osm_id  # node 轉 relation
        else:
            st.error("無法處理的 OSM 類型")
            return pd.DataFrame()
    except GeocoderServiceError:
        st.error("地理編碼服務出錯，請稍後再試")
        return pd.DataFrame()

    # 根據類別設置 Overpass API 查詢
    if category == 'restaurant':
        query = f"""
        [out:json][timeout:25];
        area({area_id})->.searchArea;
        (
          node["amenity"="restaurant"](area.searchArea);
          way["amenity"="restaurant"](area.searchArea);
          relation["amenity"="restaurant"](area.searchArea);
        );
        out center;
        """
    elif category == 'attraction':
        query = f"""
        [out:json][timeout:25];
        area({area_id})->.searchArea;
        (
          node["tourism"~"attraction|museum|viewpoint|zoo"](area.searchArea);
          way["tourism"~"attraction|museum|viewpoint|zoo"](area.searchArea);
          relation["tourism"~"attraction|museum|viewpoint|zoo"](area.searchArea);

          node["natural"~"peak|waterfall|wood|forest|lake"](area.searchArea);
          way["natural"~"peak|waterfall|wood|forest|lake"](area.searchArea);
          relation["natural"~"peak|waterfall|wood|forest|lake"](area.searchArea);

          node["leisure"~"park|playground|garden|sports_centre"](area.searchArea);
          way["leisure"~"park|playground|garden|sports_centre"](area.searchArea);
          relation["leisure"~"park|playground|garden|sports_centre"](area.searchArea);
        );
        out center;
        """
    elif category == 'hotel':
        query = f"""
        [out:json][timeout:25];
        area({area_id})->.searchArea;
        (
          node["tourism"~"hotel|guest_house"](area.searchArea);
          way["tourism"~"hotel|guest_house"](area.searchArea);
          relation["tourism"~"hotel|guest_house"](area.searchArea);
        );
        out center;
        """
    else:
        st.error("未知的類別")
        return pd.DataFrame()

    api_url = "http://overpass-api.de/api/interpreter"
    try:
        response = requests.get(api_url, params={'data': query})
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            place_list = []
            conn = get_connection()
            cursor = conn.cursor()
            for elem in elements:
                tags = elem.get('tags', {})
                place_id = elem.get('id')
                name = tags.get('name', '未知名稱')
                # 檢查 places 資料表是否已存在該地點
                cursor.execute('SELECT * FROM places WHERE place_id = ?', (place_id,))
                place_exists = cursor.fetchone()
                if place_exists:
                    # 已存在，直接使用資料表中的地址
                    address = place_exists[2]  # address
                    latitude = place_exists[3]
                    longitude = place_exists[4]
                else:
                    # 不存在，組合地址或設為 '地址不詳'
                    address = tags.get('addr:full', '')
                    if not address:
                        # 組合地址，擴展更多地址標籤
                        addr_parts = []
                        for key in ['addr:housenumber', 'addr:street', 'addr:suburb', 'addr:city', 'addr:state', 'addr:postcode', 'addr:country']:
                            if key in tags:
                                addr_parts.append(tags[key])
                        address = ', '.join(addr_parts) if addr_parts else '地址不詳'
                    # 獲取經緯度
                    if elem['type'] == 'node':
                        latitude = elem.get('lat')
                        longitude = elem.get('lon')
                    elif 'center' in elem:
                        latitude = elem['center'].get('lat')
                        longitude = elem['center'].get('lon')
                    else:
                        continue  # 無法獲取座標，跳過
                    if latitude is None or longitude is None:
                        continue  # 座標缺失，跳過
                    # 插入 places 資料表
                    cursor.execute('''
                        INSERT INTO places (place_id, name, address, latitude, longitude, category)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (place_id, name, address, latitude, longitude, category))
                    conn.commit()
                # 構建地點列表
                place_list.append({
                    'place_id': place_id,
                    'name': name,
                    'address': address,
                    'latitude': latitude,
                    'longitude': longitude,
                    'category': category
                })
            conn.close()
            df = pd.DataFrame(place_list)
            if df.empty:
                st.warning("未從 Overpass API 獲取到任何地點數據。請確認城市名稱是否正確。")
            else:
                st.success(f"成功載入 {len(df)} 個地點。")
            return df
        else:
            st.error("地點數據載入失敗")
            return pd.DataFrame()
    except requests.RequestException as e:
        st.error(f"無法連接到 Overpass API: {e}")
        return pd.DataFrame()

# 反向地理編碼
@st.cache_data
def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent="review_app")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language='zh-TW')
        if location:
            return location.address
        else:
            return '地址不詳'
    except GeocoderServiceError:
        return '地址不詳'

# 載入評論數據（根據 place_id）
def get_reviews_by_place_id(place_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews WHERE place_id = ? AND display = 1 ORDER BY timestamp DESC', (place_id,))
    reviews = cursor.fetchall()
    conn.close()
    return reviews

# 計算用戶信用分數百分比
def calculate_user_credit_percentage():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, credit_score FROM users')
    users = cursor.fetchall()
    total_score = sum([user[1] for user in users])
    user_percentages = {}
    for user in users:
        user_id, score = user
        percentage = (score / total_score) * 100 if total_score > 0 else 0
        # 限制信用分數在0到100之間
        percentage = max(0, min(100, percentage))
        user_percentages[user_id] = percentage
    conn.close()
    return user_percentages

# 更新用戶信用分數
def update_user_credit_score(user_id, delta):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT credit_score FROM users WHERE user_id = ?', (user_id,))
    current_score = cursor.fetchone()[0]
    new_score = max(0, current_score + delta)  # 確保信用分數不低於0
    # 限制信用分數不超過100
    new_score = min(new_score, 100.0)
    cursor.execute('UPDATE users SET credit_score = ? WHERE user_id = ?', (new_score, user_id))
    conn.commit()
    conn.close()

# 加載分類模型
@st.cache_resource
def load_classification_model():
    classifier = pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')
    return classifier

classifier = load_classification_model()

# 分句函數
def split_sentences(text):
    sentences = re.split('(?<=[。！？\?])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# 句子分類函數
def classify_sentence(sentence):
    candidate_labels_re = ['理性', '感性']
    result_re = classifier(sentence, candidate_labels_re, hypothesis_template="這句話是{}的。")
    re_label = result_re['labels'][0]

    candidate_labels_sentiment = ['正面', '負面']
    result_sentiment = classifier(sentence, candidate_labels_sentiment, hypothesis_template="這句話的情感是{}的。")
    sentiment_label = result_sentiment['labels'][0]

    return {
        'sentence': sentence,
        'sentiment': sentiment_label,
        'sentiment_score': result_sentiment['scores'][0],
        'rational_emotional': re_label,
        're_score': result_re['scores'][0],
        'category': f"{re_label}{sentiment_label}"
    }

# 過濾句子函數，只過濾「感性負面」
def filter_sentences(classified_sentences):
    filtered_sentences = [s['sentence'] for s in classified_sentences if s['category'] != '感性負面']
    return filtered_sentences

# 檢查是否包含感性負面內容
def contains_emotional_negative(classified_sentences):
    for s in classified_sentences:
        if s['category'] == '感性負面':
            return True
    return False

# 重新生成評論函數
def regenerate_review(filtered_sentences):
    return ''.join(filtered_sentences)

# 保存評論函數
def save_review(place_id, review_text, rating, anonymous, display):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if anonymous:
        user_id = None  # 匿名用戶不設置 user_id
    else:
        user_id = st.session_state['user_id']
    try:
        cursor.execute('''
            INSERT INTO reviews (place_id, user_id, review_text, rating, timestamp, display)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (place_id, user_id, review_text, rating, timestamp_now, display))
        conn.commit()
        st.success("評論已提交")
    except sqlite3.OperationalError as e:
        st.error(f"評論提交失敗：{e}")
    finally:
        conn.close()

def analyze_review(review_text):
    s = SnowNLP(review_text)
    sentiment_score = s.sentiments  # 0 到 1 之間的分數
    if sentiment_score > 0.6:
        sentiment = '正面'
    elif sentiment_score < 0.4:
        sentiment = '負面'
    else:
        sentiment = '中立'
    return sentiment, sentiment_score

def predict_rating(sentiment, score):
    if sentiment == '正面':
        if score >= 0.8:
            rating = 5
        elif score >= 0.6:
            rating = 4
        else:
            rating = 3  # 即使是正面，但分數較低時
    elif sentiment == '負面':
        if score <= 0.2:
            rating = 1
        elif score <= 0.4:
            rating = 2
        else:
            rating = 3  # 即使是負面，但分數較高時
    else:
        rating = 3  # 中立
    return rating

# 計算地點整體評分
def calculate_place_overall_rating(place_id):
    reviews = get_reviews_by_place_id(place_id)
    if not reviews:
        return None
    user_credit_percentages = calculate_user_credit_percentage()
    total_weighted_score = 0
    total_weight = 0
    for review in reviews:
        user_id = review[2]
        rating = review[4]
        if user_id and user_id in user_credit_percentages:
            weight = user_credit_percentages[user_id]
        else:
            weight = 1  # 匿名用戶或無信用分數的默認權重為1
        total_weighted_score += rating * weight
        total_weight += weight
    if total_weight == 0:
        return None
    overall_rating = total_weighted_score / total_weight
    return round(overall_rating, 1)

# 顯示地點詳細信息
def display_place_details(place):
    st.subheader(f"{place['category'].capitalize()}：{place['name']}")
    address = place.get('address', '地址不詳')

    # 如果地址是 '地址不詳'，則進行反向地理編碼
    if address == '地址不詳':
        st.write("地址：正在獲取地址資訊...")
        address = reverse_geocode(place['latitude'], place['longitude'])
        # 更新 places 資料表
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE places SET address = ? WHERE place_id = ?', (address, place['place_id']))
        conn.commit()
        conn.close()
        st.write(f"地址：{address}")
    else:
        st.write(f"地址：{address}")

    # 顯示地點整體評分
    overall_rating = calculate_place_overall_rating(place['place_id'])
    if overall_rating:
        st.write(f"整體評分：{overall_rating} 星")
    else:
        st.write("尚無評分")

    # 顯示地圖位置
    if place.get('latitude') and place.get('longitude'):
        st.write("地點位置：")
        df_map = pd.DataFrame({
            'lat': [place['latitude']],
            'lon': [place['longitude']]
        })
        st.map(df_map)
    else:
        st.write("無法顯示地圖位置")

# 顯示評論並標註用戶信用分數
def display_reviews(place_id):
    reviews = get_reviews_by_place_id(place_id)
    st.subheader("評論")
    if reviews:
        user_credit_percentages = calculate_user_credit_percentage()
        for row in reviews:
            # 假設 reviews 的字段順序為：
            # review_id, place_id, user_id, review_text, rating, timestamp, display
            user_id = row[2]
            review_text = row[3]
            rating = row[4]
            timestamp = row[5]
            if user_id:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT username FROM users WHERE user_id = ?', (user_id,))
                user_info = cursor.fetchone()
                conn.close()
                username = user_info[0] if user_info else '匿名用戶'
                user_score = user_credit_percentages.get(user_id, 0.0)
            else:
                username = '匿名用戶'
                user_score = 0.0
            st.markdown(f"**用戶：{username}（信用分數：{user_score:.2f}%）**")
            st.markdown(f"評分：{rating} 星")
            st.markdown(f"評論內容：{review_text}")
            st.markdown(f"時間：{timestamp}")
            st.markdown("---")
    else:
        st.write("暫無評論")

# 添加收藏功能
def add_to_favorites(user_id, place):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM favorites WHERE user_id = ? AND place_id = ?', (user_id, place['place_id']))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute('''
            INSERT INTO favorites (user_id, place_id)
            VALUES (?, ?)
        ''', (user_id, place['place_id']))
        conn.commit()
        st.success("已添加到收藏")
    else:
        st.info("該地點已在您的收藏中")
    conn.close()

# 移除收藏功能
def remove_from_favorites(user_id, place_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM favorites WHERE user_id = ? AND place_id = ?', (user_id, place_id))
    conn.commit()
    st.success("已從收藏中移除")
    conn.close()

# 顯示用戶收藏的地點
def display_favorites(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT places.place_id, places.name, places.address, places.latitude, places.longitude, places.category
        FROM favorites
        JOIN places ON favorites.place_id = places.place_id
        WHERE favorites.user_id = ?
    ''', (user_id,))
    favorites = cursor.fetchall()
    conn.close()
    st.subheader("我的收藏")
    if favorites:
        for fav in favorites:
            st.write(f"**{fav[5].capitalize()}：{fav[1]}**")
            st.write(f"地址：{fav[2]}")
            if st.button("移除收藏", key=f"remove_{fav[0]}"):
                remove_from_favorites(user_id, fav[0])
                # st.experimental_rerun()  # 重新載入頁面以更新收藏列表
    else:
        st.write("您還沒有收藏任何地點。")

# 提交評論
def submit_review(place):
    st.subheader("發表評論")

    # 初始化表單計數器
    if f"review_form_counter_{place['place_id']}" not in st.session_state:
        st.session_state[f"review_form_counter_{place['place_id']}"] = 0

    form_counter_key = f"review_form_counter_{place['place_id']}"
    form_key = f"submit_review_form_{place['place_id']}_{st.session_state[form_counter_key]}"
    text_area_key = f"review_text_{place['place_id']}_{st.session_state[form_counter_key]}"
    checkbox_key = f"anonymous_{place['place_id']}_{st.session_state[form_counter_key]}"

    with st.form(key=form_key, clear_on_submit=True):
        review_text = st.text_area("評論內容", key=text_area_key)
        anonymous = st.checkbox("匿名發表", key=checkbox_key)
        submit_button = st.form_submit_button(label="提交評論")

        if submit_button:
            if review_text.strip():  # 使用 strip() 檢查非空白內容
                # 分句
                sentences = split_sentences(review_text)
                classified_sentences = [classify_sentence(s) for s in sentences]
                has_emotional_negative = contains_emotional_negative(classified_sentences)
                if has_emotional_negative:
                    # 過濾感性負面的句子
                    filtered_sentences = filter_sentences(classified_sentences)
                    new_review = regenerate_review(filtered_sentences)
                    # 如果過濾後的評論為空
                    if not new_review.strip():
                        st.error("您的評論包含過多感性負面內容，無法提交，信用分數將降低。")
                        # 降低用戶信用分數
                        if not anonymous:
                            update_user_credit_score(st.session_state['user_id'], -2)
                    else:
                        # 將新評論顯示給用戶並要求確認
                        st.session_state['new_review'] = new_review
                        st.session_state['anonymous'] = anonymous
                        st.session_state['place_id'] = place['place_id']
                else:
                    # 無需過濾，直接提交
                    sentiment, score = analyze_review(review_text)
                    st.write(f"情感：{sentiment}, 分數：{score}")  # 添加此行以顯示情感分析結果
                    predicted_rating = predict_rating(sentiment, score)
                    st.write(f"預測評分：{predicted_rating} 星")  # 顯示預測評分
                    # 保存到資料庫
                    save_review(place['place_id'], review_text, predicted_rating, anonymous, display=1)
                    # 增加用戶信用分數
                    if not anonymous:
                        update_user_credit_score(st.session_state['user_id'], 3)
                    # 增加表單計數器以重置表單
                    st.session_state[form_counter_key] += 1
            else:
                st.error("評論內容不能空白")

    # 如果有新的評論需要確認
    if 'new_review' in st.session_state and st.session_state.get('place_id') == place['place_id']:
        st.write("### 您的評論包含感性負面的內容，已為您過濾：")
        st.write(st.session_state['new_review'])
        if st.button("確認並提交過濾後的評論"):
            new_review = st.session_state['new_review']
            anonymous = st.session_state['anonymous']
            # 重新分析過濾後的評論
            sentiment, score = analyze_review(new_review)
            predicted_rating = predict_rating(sentiment, score)
            st.write(f"情感：{sentiment}, 分數：{score}")  # 顯示情感分析結果
            st.write(f"預測評分：{predicted_rating} 星")  # 顯示預測評分
            # 保存到資料庫
            save_review(place['place_id'], new_review, predicted_rating, anonymous, display=1)
            # 增加用戶信用分數
            if not anonymous:
                update_user_credit_score(st.session_state['user_id'], 2)
            # 清除暫存的評論
            del st.session_state['new_review']
            del st.session_state['anonymous']
            del st.session_state['place_id']
            # 增加表單計數器以重置表單
            form_counter_key = f"review_form_counter_{place['place_id']}"
            st.session_state[form_counter_key] += 1

# 用戶登入或註冊頁面
def login_page():
    st.title("登入或註冊")
    username = st.text_input("名稱")
    password = st.text_input("密碼", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("登入"):
            if username and password:
                user_id = login_user(username, password)
                if user_id:
                    st.success("登入成功！")
                    st.session_state['user_id'] = user_id
                    st.session_state['username'] = username
                else:
                    st.error("名稱或密碼錯誤")
            else:
                st.error("請填寫名稱和密碼")
    with col2:
        if st.button("註冊"):
            if username and password:
                if register_user(username, password):
                    st.success("註冊成功，請登入")
                else:
                    st.error("名稱已存在")
            else:
                st.error("請填寫名稱和密碼")

# 主應用程序
def main():
    if 'user_id' not in st.session_state:
        login_page()
    else:
        # 側邊欄
        st.sidebar.header("選單")
        menu_options = ["搜尋地點", "我的收藏"]
        choice = st.sidebar.selectbox("功能選單", menu_options)

        st.sidebar.write(f"歡迎，{st.session_state['username']}！")

        # 計算用戶信用分數百分比
        user_credit_percentages = calculate_user_credit_percentage()
        user_id = st.session_state['user_id']
        credit_score = user_credit_percentages.get(user_id, 0.0)
        st.sidebar.write(f"信用分數：{credit_score:.2f}%")

        if choice == "搜尋地點":
            # 側邊欄搜索功能
            st.sidebar.header("搜尋功能")
            city_list = [
                "台北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣",
                "苗栗縣", "台中市", "彰化縣", "南投縣", "雲林縣", "嘉義市",
                "嘉義縣", "台南市", "高雄市", "屏東縣", "宜蘭縣", "花蓮縣",
                "台東縣", "澎湖縣", "金門縣", "連江縣"
            ]
            selected_city = st.sidebar.selectbox("選擇縣市", city_list)
            category_list = ["餐廳", "旅遊景點", "旅館住宿"]
            category_mapping = {"餐廳": "restaurant", "旅遊景點": "attraction", "旅館住宿": "hotel"}
            selected_category_display = st.sidebar.selectbox("選擇類別", category_list)
            selected_category = category_mapping[selected_category_display]
            search_term = st.sidebar.text_input("搜尋地點（名稱）")

            # 主頁面標題
            st.title("綜合評論平台")

            # 獲取城市經緯度
            city_lat, city_lon = get_city_coordinates(selected_city)

            if city_lat is None or city_lon is None:
                st.stop()  # 無法獲取城市座標，停止執行

            # 載入地點數據（根據城市和類別）
            cache_key = f"all_places_{selected_city}_{selected_category}"
            if (cache_key not in st.session_state):
                df_places = load_place_data_by_city_and_category(selected_city, selected_category)
                st.session_state[cache_key] = df_places
            else:
                df_places = st.session_state[cache_key]

            # 應用搜索過濾條件
            if search_term and not df_places.empty:
                df_search_results = df_places[df_places['name'].str.contains(search_term, case=False, na=False)]
            else:
                df_search_results = df_places.copy()

            # 顯示搜索結果列表
            st.subheader("地點列表")
            if not df_search_results.empty:
                place_names = df_search_results['name'].unique().tolist()
                selected_place_name = st.selectbox("選擇地點查看詳細資訊", [""] + place_names)
                if selected_place_name:
                    selected_place = df_search_results[df_search_results['name'] == selected_place_name].iloc[0]
                    display_place_details(selected_place)

                    # 收藏按鈕
                    if st.button("加入收藏", key=f"favorite_{selected_place['place_id']}"):
                        add_to_favorites(st.session_state['user_id'], selected_place)
                        # st.experimental_rerun()  # 重新載入頁面以更新收藏狀態

                    display_reviews(selected_place['place_id'])
                    submit_review(selected_place)
            else:
                if search_term:
                    st.write("未找到相關地點")
                else:
                    st.write("請在側邊欄輸入關鍵字搜尋地點")
        elif choice == "我的收藏":
            st.title("我的收藏")
            display_favorites(st.session_state['user_id'])

        # 登出按鈕
        if st.sidebar.button("登出"):
            st.session_state.clear()
            st.success("已登出")

if __name__ == "__main__":
    main()