from flask import Flask, render_template, request , jsonify, session, app, Response, redirect, url_for,flash
from utils.actions import login_validation_check , selling_injection_in_mongo , generate_response , signup_mongo ,compute_plan_agri , apple_count , weed_detection , leaf_disease_detection , fetch_store_documents,scrape_agriculture_news, get_weather
import os
from pymongo import MongoClient
import ollama
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import base64
from ultralytics import YOLO
import numpy as np
import cv2
import requests

client = MongoClient("mongodb://localhost:27017/")  # Ensure MongoDB is running
db = client["agribot"]

PROTECTED_ROUTES = [
    '/homepage', '/storepage', '/sellingpage', '/financialpage', 
    '/Insurancepage', '/loanpage', '/loanformpage', '/insuranceformpage', 
    '/chatbotpage', '/dashboardpage', '/communicationpage', '/newspage', 
    '/quickstartpage', '/cvpage'
]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

app.secret_key = "summasecretkey"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

@app.before_request
def before_request():
    # Check if the requested endpoint is in the protected routes
    if request.path in PROTECTED_ROUTES:
        # Check if the user is logged in
        if 'number' not in session or 'type' not in session:
            return redirect(url_for('hello_world')) 
        

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/logout")
def logout():
    # Clear the session
    session.clear()
    # Redirect to the home page
    return redirect(url_for('hello_world'))

@app.route("/farmerlogin")
def farmerlogin():
    return render_template("farmerlogin.html")

@app.route("/buyerlogin")
def buyerlogin():   
    return render_template("buyerlogin.html")

@app.route("/farmerloginauth", methods=["POST"])
def farmerloginauth():
    number = request.form.get("number")
    password = request.form.get("password")
    validation_result = login_validation_check(number, password,"farmer")
    session['number']=number
    session['type']="farmer"
    if validation_result:
        return redirect("/homepage")
    else:
        flash("Wrong number or password, try again", "error")  # Flash an error message
        return render_template("farmerlogin.html") 
    
@app.route('/homepage') 
def homepage():
    if 'number' not in session or 'type' not in session:
        return redirect(url_for('hello_world'))  # Redirect if not logged in

    # Fetch the user's name
    user_name = get_user_name(session['number'], session['type'])
    if not user_name:
        return "User not found", 404

    return render_template("homepage.html", user_name=user_name)



@app.route("/buyerloginauth", methods=["POST"])
def buyerloginauth():
    number = request.form.get("number")
    password = request.form.get("password")
    validation_result = login_validation_check(number, password, "buyer")
    session['number']=number
    session['type']="buyer"
    if validation_result:
        return redirect("/homepage")
    else:
        flash("Wrong number or password, try again", "error")  # Flash an error message
        return render_template("buyerlogin.html") 

@app.route("/buyersignup")
def buyersignup():
    return render_template("signup.html")

@app.route("/farmersignup")
def farmersignup():
    return render_template("signupfarmer.html")  

@app.route("/signupprocessfarmer", methods=["POST"])
def signupprocessfarmer():
    name = request.form.get('fname') + " " + request.form.get('lname')
    mobile_number = (request.form.get('mobileno'))
    password = request.form.get('password')
    confirm_password=request.form.get('confirm_password')
    address = request.form.get('address')
    gender = request.form.get('gender')
    age = int(request.form.get('age'))
    dateofbirth = request.form.get('dob')
    email = request.form.get('email')
    blood_group = request.form.get('bloodgroup')
    unique_id = request.form.get('aadhaar')
    district=request.form.get('district')
    state = request.form.get('state')
    country = request.form.get('country')
    type="farmer"

    

    if password != confirm_password:
        return jsonify({"message": "Passwords do not match!"}), 400

    signup_mongo(name, mobile_number, password, address, gender, age, dateofbirth, email, blood_group, unique_id, district, state, country,type)

    session["type"]=type
    session["number"]=mobile_number
    return render_template("farmerlogin.html")

@app.route("/signupprocessbuyer", methods=["POST"])
def signupprocessbuyer():
    name = request.form.get('fname') + " " + request.form.get('lname')
    mobile_number = (request.form.get('mobileno'))
    password = request.form.get('password')
    confirm_password=request.form.get('confirm_password')
    address = request.form.get('address')
    gender = request.form.get('gender')
    age = int(request.form.get('age'))
    dateofbirth = request.form.get('dob')
    email = request.form.get('email')
    blood_group = request.form.get('bloodgroup')
    unique_id = request.form.get('aadhaar')
    district=request.form.get('district')
    state = request.form.get('state')
    country = request.form.get('country')
    type="buyer"

    if password != confirm_password:
        return jsonify({"message": "Passwords do not match!"}), 400

    signup_mongo(name, mobile_number, password, address, gender, age, dateofbirth, email, blood_group, unique_id, district, state, country,type)

    session["type"]=type
    session["number"]=mobile_number

    return render_template("buyerlogin.html")

@app.route("/dashboard")
def dashboard():
    
    if 'number' not in session or 'type' not in session:
        return redirect(url_for('hello_world'))  # Redirect if not logged in
    
    

    user_type = session['type']
    number = session['number']



    # Fetch basic user details
    if user_type == "farmer":
        user_details = db["farmer_details"].find_one({"mobile_number": number})
        additional_details = db["farmer_dashboard"].find_one({"mobile_number": number})
        user_products = list(db["store"].find({"contact_number": number}))
        profile_picture = user_details.get("profile_picture", None)
        if not profile_picture:
            profile_picture = "D:\AgriHub-main\static\images\default-avatar-profile-icon-social-media-user-image-gray-avatar-icon-blank-profile-silhouette-vector-illustration_561158-3383.avif"
    elif user_type == "buyer":
        user_details = db["buyer_details"].find_one({"mobile_number": number})
        additional_details = db["buyer_dashboard"].find_one({"mobile_number": number})
        user_products = list(db["store"].find({"contact_number": number}))
        profile_picture = user_details.get("profile_picture", None)
        if not profile_picture:
            profile_picture = "D:\AgriHub-main\static\images\default-avatar-profile-icon-social-media-user-image-gray-avatar-icon-blank-profile-silhouette-vector-illustration_561158-3383.avif"


    

    
    else:
        return "Invalid user type", 400

    if not user_details:
        return "User not found", 404

    return render_template("dashboard.html", user_details=user_details, additional_details=additional_details, user_type=user_type, products=user_products, profile_picture=profile_picture)

@app.route("/update_profile", methods=["POST"])
def update_profile():
    if 'number' not in session or 'type' not in session:
        return redirect(url_for('hello_world'))  # Redirect if not logged in

    user_type = session['type']
    number = session['number']

    # Collect form data
    land_details = {
        "soil_type": request.form.get("soil_type"),
        "acres": request.form.get("acres"),
        "fertilizers": request.form.get("fertilizers"),
        "livestock": request.form.get("livestock"),
        "irrigation_method": request.form.get("irrigation_method"),

        "crop_type": request.form.get("crop_type"),  # Primary crops grown  
    "season": request.form.get("season"),  # Current farming season (e.g., Kharif, Rabi)  
    "water_source": request.form.get("water_source"),  # Water source (e.g., well, canal, rain-fed)  
    "organic_farming": request.form.get("organic_farming"),  # Yes/No for organic farming  
    "pesticides_used": request.form.get("pesticides_used"),  # Types of pesticides used  
    "machinery_used": request.form.get("machinery_used"),  # Farming equipment used  
    "yield_per_acre": request.form.get("yield_per_acre"),  # Estimated yield per acre  
    "market_access": request.form.get("market_access"),  # Nearest market for selling produce  
    "labor_availability": request.form.get("labor_availability"), 
    "weather_conditions": request.form.get("weather_conditions"),
    }

    # Update the respective dashboard collection
    if user_type == "farmer":
        db["farmer_dashboard"].update_one(
            {"mobile_number": number},
            {"$set": land_details},
            upsert=True  # Insert if the document doesn't exist
        )
    elif user_type == "buyer":
        db["buyer_dashboard"].update_one(
            {"mobile_number": number},
            {"$set": land_details},
            upsert=True
        )

    flash("Profile updated successfully!", "success")
    return redirect(url_for('dashboard'))

@app.route("/update_profile_picture", methods=["POST"])
def update_profile_picture():
    if 'number' not in session or 'type' not in session:
        return redirect(url_for('hello_world'))  # Redirect if not logged in

    user_type = session['type']
    number = session['number']

    if 'profile_picture' not in request.files:
        flash("No file selected", "error")
        return redirect(url_for('dashboard'))

    file = request.files['profile_picture']
    if file.filename == '':
        flash("No file selected", "error")
        return redirect(url_for('dashboard'))

    if file:
        # Save the file to the upload folder
        filename = f"{number}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Update the profile picture path in MongoDB
        if user_type == "farmer":
            db["farmer_details"].update_one(
                {"mobile_number": number},
                {"$set": {"profile_picture": filepath}}
            )
        elif user_type == "buyer":
            db["buyer_details"].update_one(
                {"mobile_number": number},
                {"$set": {"profile_picture": filepath}}
            )

        flash("Profile picture updated successfully!", "success")
        return redirect(url_for('dashboard'))

@app.route("/newspage")
def newspage():
    scrape_agriculture_news()
    collection = db.news
    city = request.args.get("city")
    # Fetch all news articles from MongoDB
    news_articles = list(collection.find({}, {"_id": 0})) 
    weather_data = get_weather(city) if city else None  # Exclude _id for clean JSON
    user_name = get_user_name(session['number'], session['type'])
    return render_template("news1.html", news_articles=news_articles,weather_data=weather_data, city=city,user_name=user_name) 

@app.route("/communicationpage")
def communicationpage():
    return render_template("communication.html") 

@app.route("/storepage", methods=["GET"])
def storepage():
    search_term = request.args.get('search', '')
    category_filter = request.args.get('category', 'all')
    sort_by = request.args.get('sort', 'default')

    products = fetch_store_documents()
    user_name = get_user_name(session['number'], session['type'])
    # Ensure image_path is properly formatted
    for product in products:
        if 'image_path' in product and product['image_path']:  # Check if image exists
            product['image_path'] = product['image_path']
        else:
            product['image_path'] = '/static/uploads/default.jpg'  # Fallback image

        # Ensure price is stored as a float for proper sorting
        try:
            product['price'] = float(product['price'])
        except ValueError:
            product['price'] = 0.0  # If conversion fails, set default price

    # Apply category filter
    if category_filter != 'all':
        products = [p for p in products if p['product_category'].lower() == category_filter.lower()]

    # Apply search filter
    if search_term:
        products = [
            p for p in products if search_term.lower() in p['product_name'].lower() 
            or search_term.lower() in p['description'].lower()
        ]

    # Apply sorting
    if sort_by == 'price-low':
        products.sort(key=lambda p: p['price'])
    elif sort_by == 'price-high':
        products.sort(key=lambda p: p['price'], reverse=True)
    elif sort_by == 'name-a-z':
        products.sort(key=lambda p: p['product_name'].lower())
    elif sort_by == 'name-z-a':
        products.sort(key=lambda p: p['product_name'].lower(), reverse=True)

    return render_template("store.html", products=products, search_term=search_term, category_filter=category_filter, sort_by=sort_by,user_name=user_name)




@app.route("/quickstartpage")
def quickstartpage():
    user_name = get_user_name(session['number'], session['type'])
    return render_template("quickfarm.html",user_name=user_name)

@app.route("/compute_plan", methods=['POST'])
def compute_plan():
    if "number" not in session or "type" not in session:
        return "User not logged in", 401  # Unauthorized access

    mobile_number = str(session["number"])  # Ensure it's a string
    user_type = session["type"]  
    landMeasurements = request.form.get("landMeasurements")
    budget = request.form.get("budget")
    machinery = request.form.get("machinery")
    labours = request.form.get("labours")
    soilType = request.form.get("soilType")
    irrigationMethod = request.form.get("irrigationMethod")
    storageFacilities = request.form.get("storageFacilities")
    waterAvailability = request.form.get("waterAvailability")
    waterQuantity = request.form.get("waterQuantity")
    farmingType = request.form.get("farmingType")
    current_month = datetime.now().strftime("%B")

    collection = db["farmer_details"] if user_type == "farmer" else db["buyer_details"]

    # Fetch user details from MongoDB
    user_data = collection.find_one({"mobile_number": mobile_number})
    district=user_data.get("district","")
    state=user_data.get("state","")
    country=user_data.get("country","")

    agri_plan = compute_plan_agri(landMeasurements, budget, machinery, labours, soilType, irrigationMethod, storageFacilities)

    user_name = get_user_name(session['number'], session['type'])
    return render_template("chatbot_response2.html",
                           landMeasurements=landMeasurements,
                           budget=budget,
                           machinery=machinery,
                           labours=labours,
                           soilType=soilType,
                           irrigationMethod=irrigationMethod,
                           storageFacilities=storageFacilities,
                           waterAvailability=waterAvailability,
                           waterQuantity=waterQuantity,
                           farmingType=farmingType,
                           current_month=current_month,
                           district=district,
                           state=state,
                           country=country,
                           agri_plan=agri_plan,
                           user_name=user_name)

@app.route("/stream_plan")
def stream_plan():
    landMeasurements = request.args.get("landMeasurements")
    budget = request.args.get("budget")
    machinery = request.args.get("machinery")
    labours = request.args.get("labours")
    soilType = request.args.get("soilType")
    irrigationMethod = request.args.get("irrigationMethod")
    storageFacilities = request.args.get("storageFacilities")
    waterAvailability = request.args.get("waterAvailability")
    waterQuantity = request.args.get("waterQuantity")
    farmingType = request.args.get("farmingType")
    current_month = request.args.get("current_month")
    district = request.args.get("district")
    state = request.args.get("state")
    country = request.args.get("country")
    
    def stream_response():
        prompt = f"""
        You are an advanced agricultural AI assistant. Generate a detailed agricultural work plan based on the following details:
        - **Current Month:** {current_month}
        - **Location:** {district}, {state}, {country}
        - **Land Measurements:** {landMeasurements} acres
        - **Budget:** {budget} INR
        - **Machinery Available:** {machinery}
        - **Number of Labors:** {labours}
        - **Soil Type:** {soilType}
        - **Irrigation Method:** {irrigationMethod}
        - **Water Availability:** {waterAvailability}
        - **Average Water Quantity Provided:** {waterQuantity} liters
        - **Storage Facilities:** {storageFacilities}
        - **Farming Type:** {farmingType} (Organic/Non-Organic)
        
        Based on the given details:
        1. Recommend the best **crops** suitable for the season and soil.
        2. Provide a **step-by-step timeline** for sowing, irrigation, fertilization, and harvesting.
        3. Suggest **resource allocation** for optimizing budget and manpower.
        4. Highlight **best agricultural practices** for higher yield.
        5. Offer **market insights** on expected profitability and storage strategies.
        
        Ensure the plan is **well-structured, practical, and region-specific**. Format your response using proper Markdown with headers (##, ###), bullet points (- or *), and numbered lists (1., 2., etc.). Do not use asterisks for emphasis, use proper Markdown formatting instead.
        """
        
        stream = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}], stream=True)
        buffer = ""
        
        for chunk in stream:
            text_chunk = chunk.get("message", {}).get("content", "")
            if text_chunk:
                # Accumulate text to properly format it
                buffer += text_chunk
                
                # Process complete sentences or paragraphs
                if "." in buffer or "\n" in buffer:
                    parts = buffer.split(".")
                    for i in range(len(parts) - 1):
                        formatted_part = parts[i].strip()
                        
                        # Clean up any formatting issues
                        formatted_part = formatted_part.replace(" -", "\n-")
                        formatted_part = formatted_part.replace("* ", "- ")
                        formatted_part = formatted_part.replace("**", "")
                        formatted_part = formatted_part.replace("*****", "- ")
                        
                        # Ensure proper indentation for lists
                        if formatted_part.startswith("1 "):
                            formatted_part = "1. " + formatted_part[2:]
                        elif formatted_part.startswith("2 "):
                            formatted_part = "2. " + formatted_part[2:]
                        elif formatted_part.startswith("3 "):
                            formatted_part = "3. " + formatted_part[2:]
                        elif formatted_part.startswith("4 "):
                            formatted_part = "4. " + formatted_part[2:]
                        elif formatted_part.startswith("5 "):
                            formatted_part = "5. " + formatted_part[2:]
                        
                        yield formatted_part + ".\n"
                        time.sleep(0.08)
                    
                    # Keep the last part in the buffer
                    buffer = parts[-1]
        
        # Yield any remaining text in the buffer
        if buffer:
            yield buffer
    
    return Response(stream_response(), content_type="text/plain")

@app.route("/sellingpage")
def sellingpage():
    if "number" not in session or "type" not in session:
        return "User not logged in", 401  # Unauthorized access

    mobile_number = str(session["number"])  # Ensure it's a string
    user_type = session["type"]  


    # Determine the correct collection
    collection = db["farmer_details"] if user_type == "farmer" else db["buyer_details"]

    # Fetch user details from MongoDB
    user_data = collection.find_one({"mobile_number": mobile_number})

    if not user_data:
        return "User details not found", 404  # Handle case where user doesn't exist

    # Extract relevant details
    user_details = {
        "name": user_data.get("name", ""),
        "email": user_data.get("email", ""),
        "contact": user_data.get("mobile_number", ""),  # Stored as string
        "address": user_data.get("address", "")
    }

    # Render template with user details pre-filled
    user_name = get_user_name(session['number'], session['type'])
    return render_template("selling_page.html",user_name=user_name,**user_details)
    
@app.route("/financialpage")
def financialpage():
    user_name = get_user_name(session['number'], session['type'])
    return render_template("financial.html",user_name=user_name)

@app.route("/Insurancepage")
def Insurancepage():
    user_name = get_user_name(session['number'], session['type'])
    return render_template("insurance.html",user_name=user_name)

@app.route("/loanpage")
def loanpage():
    user_name = get_user_name(session['number'], session['type'])
    return render_template("loan.html",user_name=user_name)




@app.route("/chatbotpage")
def chatbotpage():
    user_name = get_user_name(session['number'], session['type'])
    return render_template("new_chatbot.html",user_name=user_name)




@app.route("/sellingprocess", methods=["POST"])
def sellingprocess():
    name = request.form["name"]
    email = request.form["email"]
    contact = request.form["contact"]
    address = request.form["address"]
    product_name = request.form["productName"]
    product_type = request.form["productType"]
    quantity = request.form["quantity"]
    price = request.form["price"]
    description = request.form["description"]
    
    image_paths = []  # Store multiple image paths

    user_name = get_user_name(session['number'], session['type'])

    if "productImages" in request.files:
        files = request.files.getlist("productImages")  # Get multiple images
        for file in files:
            if file.filename != "":
                filename = file.filename
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                image_paths.append(f"/static/uploads/{filename}")  # Store file path

    # Call MongoDB insertion function (ensure it supports image_paths as a list)
    if selling_injection_in_mongo(name, email, contact, address, product_name, 
                                  product_type, quantity, price, description, image_paths):
        return render_template("confirmation_post.html", name=name, email=email, 
                               contact=contact, address=address, product_name=product_name, 
                               product_type=product_type, quantity=quantity, 
                               price=price, description=description, image_paths=image_paths,user_name=user_name)
    

@app.route("/update_product", methods=["POST"])
def update_product():
    updated_data = {
        "name": request.form["name"],
        "email": request.form["email"],
        "contact": request.form["contact"],
        "address": request.form["address"],
        "product_name": request.form["product_name"],
        "product_type": request.form["product_type"],
        "quantity": request.form["quantity"],
        "price": request.form["price"],
        "description": request.form["description"],
    }

    collection = db["store"]
    

    collection.update_one({"email": updated_data["email"], "product_name": updated_data["product_name"]}, {"$set": updated_data})

    return render_template("homepage.html")


@app.route("/cvpage")
def cvpage():
    user_name = get_user_name(session['number'], session['type'])
    return render_template("cv.html",user_name=user_name)

@app.route("/upload", methods=['GET', 'POST'])
def upload_page():
    task = request.args.get("task")
    result = None
    image_path = None
    user_name = get_user_name(session['number'], session['type'])
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        elif request.form.get("imageData"):
            image_data = request.form["imageData"]
            image_binary = base64.b64decode(image_data.split(",")[1])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "captured_photo.png")
            with open(filepath, "wb") as f:
                f.write(image_binary)
        else:
            return "No valid file or camera input", 400

        # Process image and get the result
        if task == "leaf":
            result, image_path = leaf_disease_detection(filepath)
        elif task == "weed":
            result, image_path = weed_detection(filepath)
        elif task == "count":
            result,image_path = apple_count(filepath)
        else:
            result = ["Invalid task!"]

        # Debugging: Print the result type
        print(f"DEBUG: Result Type: {type(result)}, Result Value: {result}")

        # Ensure result is a list before joining
        if result is None:
            result = ["No detection result"]
        elif isinstance(result, str):  # Convert a single string to a list
            result = [result]
        elif not isinstance(result, (list, tuple)):  # Ensure it's iterable
            result = [str(result)]


    return render_template("upload_form.html", task=task, result=result, image_path=image_path,user_name=user_name)


models = {
    "leaf": YOLO("yolo/plantdiseasedetection/best.pt"),
    "weed": YOLO("yolo/weeddetection/last.pt"),
    "count": YOLO("yolo/appledetection/best.pt")
}

@app.route("/video_feed", methods=['POST'])
def video_feed():
    task = request.args.get("task")
    if task not in models:
        return jsonify({"error": "Invalid task"}), 400

    model = models[task]
    
    # Decode the frame
    frame_data = request.json.get("frame")
    if not frame_data:
        return jsonify({"error": "No frame data received"}), 400

    frame_bytes = base64.b64decode(frame_data.split(",")[1])
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model(frame)
    
    # Process results
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  # Class index
            conf = float(box.conf[0])  # Confidence
            detections.append({"class": model.names[cls], "confidence": round(conf, 2)})

    return jsonify({"prediction": detections if detections else "No detection"})


@app.route('/leafbase', methods=['GET', 'POST'])
def leafbase():
    user_name = get_user_name(session['number'], session['type'])
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Call the leaf model function here
            result = leaf_disease_detection(filepath)
            return render_template('leafresult.html', result=result)
    return render_template('upload_form.html', task='leaf',user_name=user_name)

@app.route('/weedbase', methods=['GET', 'POST'])
def weedbase():
    user_name = get_user_name(session['number'], session['type'])
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Call the leaf model function here
            result = weed_detection(filepath)
            return render_template('weedresult.html', result=result)
    return render_template('upload_form.html', task='weed',user_name=user_name)

@app.route('/countbase', methods=['GET', 'POST'])
def countbase():
    user_name = get_user_name(session['number'], session['type'])
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Call the leaf model function here
            result = apple_count(filepath)
            return render_template('countresult.html', result=result)
    return render_template('upload_form.html', task='count',user_name=user_name)



@app.route("/chatprocess", methods=["POST"])
def chatprocess():
    """
    Handle user messages and generate AI responses.
    """
    user_message = request.json.get("message", "")
    SYSTEM_PROMPT = ''' AgriMind AI  Created by Sriram

AgriMind AI is a specialized artificial intelligence designed exclusively for Agriculture and Farming. It provides accurate, relevant, and practical information on topics including agronomy, horticulture, soil science, livestock management, irrigation, pest control, crop production, and other agricultural domains.

Responses are concise and to the point. Additional details are provided only if explicitly requested.
AgriMind AI strictly answers only agriculture-related questions. If a query falls outside this scope, it will respond with: 'I couldn't help with that question since I am an Agriculture AI.'
It ensures accuracy, reliability, and relevance in every response, offering valuable insights for farmers, agronomists, and agriculture enthusiasts.
AgriMind AI does not engage in non-agricultural topics under any circumstances.'''

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Query Ollama's model
    response = ollama.chat(
        model='llama3.2:3b',  # Change model if needed
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    bot_response = response['message']['content']
    return jsonify({"response": bot_response})

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.route("/chatprocess2", methods=["POST"])
def chatprocess2():
    """
    Handle user messages and generate AI responses using Groq API with improved
    error handling, context preservation, and user session tracking.
    """
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        user_id = data.get("user_id", "default_user")  # Add user identification
        
        # Get conversation history from session or create new one
        conversation_history = session.get(f"conversation_{user_id}", [])
        
        SYSTEM_PROMPT = (
            '''AgriMind AI Created by Sriram
"AgriMind AI is a specialized artificial intelligence designed exclusively for Agriculture and Farming. It provides accurate, relevant, and practical information on topics including agronomy, horticulture, soil science, livestock management, irrigation, pest control, crop production, and other agricultural domains.
Responses are concise and to the point. Additional details are provided only if explicitly requested.
AgriMind AI strictly answers only agriculture-related questions. If a query falls outside this scope, it will respond with: 'I couldn't help with that question since I am an Agriculture AI.'
It ensures accuracy, reliability, and relevance in every response, offering valuable insights for farmers, agronomists, and agriculture enthusiasts.
AgriMind AI does not engage in non-agricultural topics under any circumstances."'''
        )
        
        if not user_message:
            return jsonify({"response": "Please enter a message."})
        
        # Prepare messages including conversation history (last 5 messages for context)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history (limited to last 5 exchanges to prevent token limits)
        for msg in conversation_history[-10:]:  # Adjust number based on your token limits
            messages.append(msg)
            
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "max_tokens": 250,
            "temperature": 1,  # Add temperature control for response variation
        }
        
        # Add timeout to prevent hanging requests
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            response_data = response.json()
            bot_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Prevent duplicate greetings by removing the first repeated sentence
            if bot_response.count("Hello.") > 1:
                bot_response = bot_response.replace("Hello.", "", 1).strip()
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": bot_response})
            
            # Save updated conversation to session
            session[f"conversation_{user_id}"] = conversation_history
            
            return jsonify({
                "response": bot_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        elif response.status_code == 429:
            # Rate limit exceeded
            return jsonify({
                "response": "I'm currently handling many requests. Please try again in a moment.",
                "error": "rate_limit"
            }), 429
            
        elif response.status_code == 401:
            # Authentication error
            app.logger.error("API Authentication failed")
            return jsonify({
                "response": "Sorry, there was an authentication issue. Please contact support.",
                "error": "auth_error"
            }), 500
            
        else:
            # Other errors
            app.logger.error(f"Groq API error: {response.status_code}, {response.text}")
            return jsonify({
                "response": "I'm having trouble processing your request. Please try again later.",
                "error": "api_error"
            }), 500
            
    except requests.Timeout:
        return jsonify({
            "response": "The request timed out. Please try again.",
            "error": "timeout"
        }), 504
        
    except Exception as e:
        app.logger.error(f"Unexpected error in chatprocess2: {str(e)}")
        return jsonify({
            "response": "An unexpected error occurred. Please try again.",
            "error": "server_error"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)