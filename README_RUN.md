# 🧠 PeatLearn AI-Enhanced Adaptive Learning System

## 🚀 Quick Start - See Everything in Action!

### **One-Command Setup & Launch:**

```bash
# Make sure you're in the PeatLearn directory
cd /Users/aban/drive/Projects/PeatLearn

# Activate virtual environment and run
source venv/bin/activate && python run_peatlearn.py
```

That's it! 🎉 This will automatically:
- ✅ Check and install all dependencies
- ✅ Set up data directories
- ✅ Check API keys
- ✅ Launch the full Streamlit dashboard

---

## 🎯 What You'll See in Action

### **1. 💬 Intelligent Chat Interface**
- Chat with Ray Peat AI about bioenergetics
- **Live AI Profiling**: Every message gets analyzed by Gemini
- **Feedback Buttons**: 👍👎 to train your personal profile
- **Adaptive Responses**: Answers get simpler/more detailed based on your level

### **2. 📊 Real-Time Profile Building** 
- **Topic Mastery**: See your levels in metabolism, hormones, stress, etc.
- **Learning Style**: AI detects if you're an "Explorer" or "Deep Diver"
- **Progress Tracking**: Visual charts of your learning journey

### **3. 💡 AI-Generated Recommendations**
- **Personalized**: Based on your exact learning pattern
- **Priority-Based**: High/Medium/Low recommendations
- **Smart Content**: "Dive deeper into CO2" or "Review thyroid basics"

### **4. 🎯 Adaptive Quizzes**
- **Difficulty Matching**: Easy for beginners, challenging for advanced
- **Topic-Focused**: Quiz on your weak areas or interests
- **Progress Integration**: Results feed back into your profile

### **5. 📈 Learning Analytics**
- **Daily Activity**: See your interaction patterns
- **Topic Distribution**: Which areas you explore most
- **Feedback Trends**: Positive/negative feedback over time

---

## 🧠 AI Features Explained

### **Smart Profiling with Gemini 2.5-Flash**
- **Question Analysis**: Rates complexity (1-5) and understanding level
- **Learning Pattern Detection**: Identifies progression and velocity  
- **Personalized Insights**: "Fast learner", "Prefers hormones topics"

### **Adaptive Content Selection**
- **Struggling Users**: Get simplified explanations and basics
- **Advanced Users**: Get detailed biochemical mechanisms
- **Learning Users**: Get balanced, progressive content

### **Example in Action:**

**Beginner asks:** "What is thyroid?"
**AI Response:** "The thyroid gland produces hormones T3 and T4 that control your metabolism..."

**Advanced user asks:** "How does T3 conversion work in peripheral tissues?"  
**AI Response:** "T3 conversion from T4 occurs via deiodinase enzymes. Ray Peat's model emphasizes optimal conversion requires cellular energy, proper liver function..."

---

## 🔧 Troubleshooting

### **If AI Features Don't Work:**
1. Check your `.env` file has: `GOOGLE_API_KEY=your_actual_key`
2. Get a free key from: https://makersuite.google.com/app/apikey
3. Restart: `python run_peatlearn.py`

### **If Dependencies Missing:**
The startup script automatically installs them, but manually:
```bash
pip install streamlit plotly pandas google-generativeai python-dotenv
```

### **If Port 8501 is Busy:**
Edit `run_peatlearn.py` and change `--server.port=8501` to another port like `8502`

---

## 🎉 Demo Flow Suggestion

1. **Start the system**: `python run_peatlearn.py`
2. **Enter your name**: Try "Demo User" 
3. **Ask basic questions**: "What is metabolism?"
4. **Give feedback**: Click 👍 or 👎 on responses
5. **Watch profile build**: Go to Profile tab, see mastery levels
6. **Ask advanced questions**: "How does T3 conversion work?"
7. **See recommendations**: Check how AI suggests next topics
8. **Try quiz**: Generate personalized quiz on your topics
9. **View analytics**: See your learning patterns visualized

---

## 📁 System Architecture

```
PeatLearn/
├── peatlearn_master.py          # Main Streamlit dashboard
├── run_peatlearn.py             # One-command startup script
├── src/adaptive_learning/        # AI-enhanced learning modules
│   ├── ai_profile_analyzer.py   # Gemini-powered profiling
│   ├── data_logger.py           # Interaction tracking
│   ├── content_selector.py      # Adaptive content
│   └── quiz_generator.py        # Personalized quizzes
├── data/user_interactions/       # User data storage
└── .env                         # API keys
```

**Ready to see your adaptive learning system in action? 🚀**

Just run: `source venv/bin/activate && python run_peatlearn.py`
