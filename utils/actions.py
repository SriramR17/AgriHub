import re
'''from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
import os'''
'''from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM , AutoTokenizer , pipeline , BitsAndBytesConfig'''
import os
import torch
import time
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import ollama
#import faiss
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient




from pymongo import MongoClient 
client = MongoClient('mongodb://localhost:27017/') 
db = client['agribot']

def login_validation_check(number, password, type):
    if type == "farmer":
        collection = db['farmer_details']  
        user_data = collection.find_one({"mobile_number": number})

    else:
        collection = db['buyer_details'] 
        user_data = collection.find_one({"mobile_number": number})
    if user_data:
        if user_data['password'] == password:
            return True
        else:
            return False
    else:
        return False 
        
def selling_injection_in_mongo(name, email, contact, address, product_name, product_type, quantity, price, description, image_path): 
    collection = db['store']  
    selling_data = {
        "name": name,
        "email": email,
        "contact_number": contact,
        "locality_address": address,
        "product_name": product_name,
        "product_quantity": quantity,
        "image_path": image_path,
        "price": price,
        "description": description,
        "product_category": product_type
    }
    try:
        collection.insert_one(selling_data)  
        print("Selling data inserted successfully!")
        return True
    except Exception as e:
        print(f"Error during insertion: {e}")
        return False

def signup_mongo(name, mobile_number, password, address, gender, age, dateofbirth, email, blood_group, unique_id, district, state, country,type):

    if type=="farmer":
        collection = db['farmer_details'] 
    else:
        collection=db["buyer_details"]

    existing_user = collection.find_one({"mobile_number": mobile_number})
    
    if existing_user:
        return {"status": "error", "message": "Mobile number already registered!"}
    
    user_data = {
        "name": name,
        "mobile_number": mobile_number,
        "password": password,
        "address": address,
        "gender": gender,
        "age": age,
        "dateofbirth": dateofbirth,
        "email": email,
        "blood_group": blood_group,
        "unique_id": unique_id,
        'district':district,
        "state": state,
        "country": country
    }
    try:
        collection.insert_one(user_data)  
        print("Signup successful!")
    except Exception as e:
        print(f"Error during signup: {e}")


import ollama

def compute_plan_agri(landMeasurements, budget, machinery, labours, soilType, irrigationMethod, storageFacilities):
    prompt = f"""
    You are an agricultural AI assistant. Based on the following details, generate a detailed agricultural work plan:

    - **Land Measurements:** {landMeasurements} in acres
    - **Budget:** {budget} in ruppees
    - **Machinery Available:** {machinery}
    - **Number of Labours:** {labours}
    - **Soil Type:** {soilType}
    - **Irrigation Method:** {irrigationMethod}
    - **Storage Facilities:** {storageFacilities}

    Provide a structured work plan, including crop selection, timeline, resource allocation, and best practices.
    """

    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])

    text= response["message"]["content"]
    cleaned_text = re.sub(r'\*+', '', text)  # Remove all occurrences of "*"
    return cleaned_text.strip()

def apple_count(video_path):
    model = YOLO(r'yolo\appledetection\best.pt')

    class_name=['Apple']
    video_path_out = '{}_out.mp4'.format(video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    output_filename = "apple_detection_result.jpg" 
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    threshold = 0.5
    max_apple_count = 0

    while ret:
        results = model(frame)[0]
        apple_count = 0

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                if int(class_id) == 0:  
                    apple_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        if apple_count > max_apple_count:
            max_apple_count = apple_count

        # Display apple count on the frame (optional)
        cv2.putText(frame, f'Apples: {apple_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        

        # Write the frame with bounding boxes and apple count
        out.write(frame)
        detected_frame = frame.copy()
        ret, frame = cap.read()

    cv2.imwrite(output_path, detected_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return max_apple_count, output_path

UPLOAD_FOLDER = "static/uploads"
# Perform inference
def leaf_disease_detection(image_path):
    class_names = [
        'Hawar_Daun', 'Virus_Kuning_Keriting', 'Hangus_Daun',
        'Defisiensi_Kalsium', 'Bercak_Daun', 'Yellow_Vein_Mosaic_Virus'
    ]

    model = YOLO(r"yolo\plantdiseasedetection\best.pt")
    img = cv2.imread(image_path)
    
    if img is None:
        return None, None  # Handle case where image is not read properly

    results = model(img)
    detected_diseases = []  # Store all detected diseases

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            label = f"{class_names[class_id]} {confidence:.2f}"
            detected_diseases.append(label)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(UPLOAD_FOLDER, "result_leaf.png")
    cv2.imwrite(output_path, img)

    return detected_diseases if detected_diseases else ["No disease detected"], output_path


def weed_detection(image_path):
    class_names = [
        "Carpetweeds", "Crabgrass", "Eclipta", "Goosegrass", "Morningglory",
        "Nutsedge", "Palmeramaranth", "Pricklysida", "Purslane", "Ragweed",
        "Sicklepod", "Spottedspurge", "Spurredanoda", "Swinecress", "Waterhemp"
    ]

    model = YOLO(r"yolo\weeddetection\last.pt")
    img = cv2.imread(image_path)
    results = model(img)

    if not results:
        return None, None

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            label = f"{class_names[class_id]} {confidence:.2f}"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed image
    output_path = os.path.join(UPLOAD_FOLDER, "result_weed.png")
    cv2.imwrite(output_path, img)

    return class_names[class_id], output_path



def fetch_store_documents():
    collection = db['store']
    documents = collection.find()
    return list(documents)


def scrape_agriculture_news( ):
    # MongoDB Connection
    max_pages=5
    url="https://economictimes.indiatimes.com/news/economy/agriculture?from=mdr"
    client = MongoClient("mongodb://localhost:27017/")  # Update if needed
    db = client.agribot  # Database: agribot
    collection = db.news  # Collection: news

    # Delete existing news collection before inserting fresh data
    collection.drop()
    print("Old news collection deleted. Starting fresh scrape...")

    page = 1
    while page <= max_pages:
        print(f"Fetching page {page}...")

        # Fetch the webpage content
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print("Failed to fetch the webpage")
            break

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all news articles in "eachStory" divs
        articles = soup.find_all('div', class_='eachStory')

        for article in articles:
            # Extract Article URL (from any <a> tag inside "eachStory")
            anchor_tag = article.find('a')
            article_url = anchor_tag['href'] if anchor_tag and anchor_tag.has_attr('href') else None
            if article_url and not article_url.startswith("http"):
                article_url = "https://economictimes.indiatimes.com" + article_url

            # Extract Image URL (from <span class="imgContainer"> > <img>)
            img_tag = article.find('span', class_='imgContainer')
            img_url = img_tag.find('img')['src'] if img_tag and img_tag.find('img') else None

            # Extract Title (from <h3> tag)
            title_tag = article.find('h3')
            title = title_tag.text.strip() if title_tag else None

            # Extract Published Date (from <time> tag)
            time_tag = article.find('time')
            published_date = time_tag.text.strip() if time_tag else None

            # Extract Description (from <p> tag)
            description_tag = article.find('p')
            description = description_tag.text.strip() if description_tag else None

            # Create a document for MongoDB
            news_document = {
                "Article URL": article_url,
                "Image URL": img_url,
                "Title": title,
                "Published Date": published_date,
                "Description": description
            }

            # Insert into MongoDB
            collection.insert_one(news_document)
            print(f"Inserted: {title}")

        # Check for pagination (Modify if there's a "Load More" button)
        next_page_link = soup.find('a', class_='next')  # Adjust class name if different
        if next_page_link and next_page_link.has_attr('href'):
            url = "https://economictimes.indiatimes.com" + next_page_link['href']
        else:
            break  # Stop if no more pages

        page += 1  # Increment page count



def get_weather(city):
    API_KEY = "sk-live-cBqopiU5Z6xpwEFdYmz60fRzfqGong7ZKfevGYHp"
    BASE_URL = "https://weather.indianapi.in/india/weather"
    url = f"{BASE_URL}?city={city}"
    headers = {"x-api-key": API_KEY}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None
    



def get_user_name(number, user_type):
    """
    Fetch the user's name from the appropriate collection based on user type.
    """
    if user_type == "farmer":
        collection = db["farmer_details"]
    elif user_type == "buyer":
        collection = db["buyer_details"]
    else:
        return None  # Invalid user type

    user = collection.find_one({"mobile_number": number})
    if user:
        return user.get("name", "User")  # Return the name or a default value
    return None


def get_profile_picture(mobile_number):
    user = db.farmer_details.find_one({"mobile_number": mobile_number}, {"profile_picture": 1}) or \
           db.buyer_details.find_one({"mobile_number": mobile_number}, {"profile_picture": 1})

    if user and "profile_picture" in user:
        # Convert backslashes to forward slashes
        return user["profile_picture"].replace("\\", "/")