# AgroVisionAI
AgroVision AI is a Multimodal AI chatbot that transforms agriculture by combining visual, textual, and audio inputs to solve complex problems.


# [Optional] Creating virtual environment
python -m venv venv
source venv/bin/activate

# Download dependencies
pip install -r requirements.txt

# Setup Gemini API key
touch .env
# Enter your Google Gemini API Key in ".env" like this:
# GOOGLE_API_KEY="..."

# Run the project
python app.py
```