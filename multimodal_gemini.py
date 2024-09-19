import google.generativeai as genai
import time
import json
import os
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="your_api_key")
from google.api_core import retry

gemini_retry = retry.Retry(
    initial=2.0,
    maximum=10.0,
    multiplier=1.0,
    deadline=60.0
)

class DermatologistBot:
    def __init__(self):
        system_instruction = "You are a Multimodal AI Chatbot designed to assist farmers with various agricultural tasks and honey bee disese detection by processing images, text, and audio inputs. Your goal is to provide accurate, timely, and actionable insights to help farmers increase productivity, manage resources efficiently, and address challenges like pest control, disease detection, and climate variability.Give the all output in the preffered language ${language}."
        self.diagnose_model = genai.GenerativeModel("models/gemini-1.5-pro-latest", system_instruction=system_instruction, generation_config={"response_mime_type": "application/json"})
        self.chat_model = genai.GenerativeModel("models/gemini-1.5-pro-latest", system_instruction=system_instruction)

        self.transcript_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        recommendation_system_prompt = """\
            Give the all output in the preffered language ${language}.
You are a Multimodal AI Chatbot designed to assist farmers with various agricultural tasks and honey bee disese detection by processing images, text, and audio inputs. Your goal is to provide accurate, timely, and actionable insights to help farmers increase productivity, manage resources efficiently, and address challenges like pest control, disease detection, and climate variability.

Core Functions:
1. Image Recognition: Analyze images of crops, soil, and livestock to detect health issues, pests, diseases, or signs of distress. Provide actionable recommendations based on the visual analysis.
2. Environmental Monitoring: Process weather, soil, and environmental data to advise farmers on optimal planting, irrigation, and harvesting times. Offer personalized insights based on the user’s location and crop type.
3. Pest and Disease Identification: Detect and diagnose pests or crop diseases from images and suggest chemical or organic treatments. Utilize a knowledge base of crop protection techniques.
4. Market Insights: Provide real-time information on crop market prices, demand, and supply chain logistics. Offer advice on the best time to sell based on trend analysis.
5. Livestock Management: Assess livestock health using video or images and monitor vital signs. Detect potential health issues through audio inputs, such as abnormal sounds from animals.
6. Interactive Support: Respond to voice or text-based queries from farmers, explaining farming techniques, best practices, or solving real-time issues.
7. Educational Content: Provide video tutorials, guides, and articles on farming techniques, sustainable agriculture, and resource management.
8. Multilingual Support: Communicate in multiple languages based on the farmer’s preference, adapting to local dialects and common agricultural terminology.

Behavior:
- Be proactive, offering suggestions based on real-time data, previous queries, and visual inputs.
- Prioritize accuracy and clarity when responding to farmers’ questions, whether they involve planting schedules, pest management, or market prices.
- Always offer follow-up options to ensure farmers receive detailed guidance if needed, such as sending more images or providing feedback on past advice.
- Remain compassionate and supportive, considering the varying levels of technological familiarity among farmers.
- Ensure low-latency responses, especially during critical periods like harvesting or disease outbreaks.

Constraints:
- Avoid providing medical advice for livestock or humans; instead, recommend contacting a veterinary or medical professional when necessary.
- When uncertain about specific crop diseases or pests, suggest collecting additional data (e.g., another image, video) or contacting an agricultural expert.
- Prioritize data privacy, ensuring any sensitive farm data shared is secure and not disclosed to third parties without permission.

Primary Goals:
- Empower farmers to make informed decisions by integrating multiple data sources (images, text, audio, environment).
- Improve yield, reduce costs, and increase sustainability in farming practices through intelligent recommendations.
- Offer solutions that are easily accessible to farmers with varying levels of education and digital literacy, including options for voice commands or simplified interfaces.


"""
        self.recommendation_model = genai.GenerativeModel("models/gemini-1.5-flash", system_instruction=recommendation_system_prompt)

        self.messages = [] # Chat history
        self.prompt_diagnose = """\
             Provide all output in the preferred language: ${language}. Give also the format heading in the preffered language ${language}.
Your farmer has uploaded an additional media to help you diagnose. Analyze the file provided and come up with a possible diagnosis and a treatment plan.
Provide the analysis in detailed paragraphs and include bullet points where necessary. And give the output format in proper spacing, line break and bold the headings.
      
"""
        return
    
    @gemini_retry
    def generate_response(self, prompt) -> str:
        self.messages.append({'role': 'user', 'parts': [prompt]})
        response = self.chat_model.generate_content(self.messages)
        self.messages.append(response.candidates[0].content)
        return response.text

    @gemini_retry
    def process_file(self, file_path) -> dict:
        
        # upload file
        file = genai.upload_file(path=file_path)

        # verify the API has successfully received the files
        while file.state.name == "PROCESSING":
            time.sleep(1)
            file = genai.get_file(file.name)

        if file.state.name == "FAILED":
            raise ValueError(file.state.name)
        
        # generate response
        prompt = self.prompt_diagnose
        self.messages.append({'role': 'user', 'parts': [file, prompt]})
        response = self.diagnose_model.generate_content(self.messages, request_options={"timeout": 60})
        self.messages.append(response.candidates[0].content)
        return json.loads(response.text)

    @gemini_retry
    def get_transcript(self, mime_type: str, audio_data: bytes) -> str:
        prompt = "Generate a transcript of the speech. If no speech transcript is available, return empty string."
        response = self.transcript_model.generate_content([
            prompt,
            {
                "mime_type": mime_type,
                "data": audio_data
            }
        ])
        return response.text.strip()
    
    @gemini_retry
    def recommand_question(self) -> str:
        prompt = f"Read the conversation history and provide a question. \nConversation history: {self.messages}"
        response = self.recommendation_model.generate_content(prompt)
        return response.text.strip()
