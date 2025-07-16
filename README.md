# 🌱 AgriHub - Cultivating Connections in Agriculture

Welcome to **AgriHub**, the digital mandi where **farmers** and **buyers** come together to sow ideas, grow connections, and reap success. Built with the finesse of **Flask**, **MongoDB**, and a frontend that even Mother Nature would swipe right on.

## 🧩 Tech Stack

- **Backend**: Flask (Python)
- **Database**:
  - MongoDB (`farmer_details`, `buyer_details`, `messages`)
- **Frontend**: HTML, CSS, JavaScript (with Tailwind for styling that’s as smooth as butter)
- **Session Management**:
  - `session['type']` → User type (`farmer` or `buyer`)
  - `session['number']` → Mobile number (used as unique ID)

## 🚜 Core Features

### 🌾 Thought Board (Community Feed)
- Users (both farmers and buyers) can post **thoughts**.
- Posts can include **text**, **images**, or both.
- Posts are shared in a communal space where others can view them.

### 👤 User Directory
- Displays a list of all **registered users** (farmers and buyers).
- Users are listed with basic details: `name`, `type`, and `mobile_number`.
- Option to **start a chat** with any user instantly.

### 💬 Real-time Chat System
- Chats are **one-to-one** and stored in MongoDB (`messages` collection).
- Supports sending and viewing messages.
- Messages are organized by:
  - `sender_number`
  - `receiver_number`
  - `timestamp`
- Each chat opens in a **dedicated chat page** for that contact.

### 🔐 Authentication Flow
- **Registration Page**:
  - Collects `name`, `mobile_number`, and `type` (`farmer` or `buyer`)
  - Saves to `farmer_details` or `buyer_details` based on type
- **Login Page**:
  - Validates credentials using mobile number and user type
- **Profile Page**:
  - Displays user info
  - Available only after login (session must be active)

## 🌟 Upcoming Features (Optional Seeds You Might Plant)
- 🌐 Language Translation for regional accessibility
- 📍 Location-based user filtering
- 🔔 Notification system for new messages
- 🏷️ Tags or categories for posts (e.g. "Fertilizer", "Market", "Weather")

## 🔧 Project Structure (Short-n-Neat)

```
AgriHub/
├── static/
│   ├── css/
│   └── js/
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   ├── feed.html
│   ├── chat.html
├── app.py
└── requirements.txt
```

## 📦 Requirements

```
Flask
pymongo
Werkzeug
```

Install with:
```bash
pip install -r requirements.txt
```

## 🚀 Running the App

```bash
python app.py
```

Open your browser and head to:
```
http://localhost:5000/
```

## 🤝 Made with ♥️ by Sriram  
_“Where agriculture meets automation, and buyers meet bhais.”_
