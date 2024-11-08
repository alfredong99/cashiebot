import os
import requests
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from pytesseract import image_to_string
from PIL import Image
import pdfplumber
from groq import Groq
from dotenv import load_dotenv
import re
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

DetectorFactory.seed = 0

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")

client = Groq(api_key=GROQ_API_KEY)

LANGUAGE_MAPPING = {
    'en': 'English',
    'zh-cn': 'Chinese (Simplified)',
    'zh': 'Chinese',
    'tl': 'Tagalog',
    'ms': 'Malay',
    'id': 'Indonesian',
    'hi': 'Hindi',
    'ta': 'Tamil'
}

LANGDETECT_TO_APP_LANG_MAP = {
    'en': 'en',
    'zh-cn': 'zh-CN',
    'zh': 'zh-CN',
    'ko': 'zh-CN',
    'tl': 'tl',
    'ms': 'ms',
    'id': 'id',
    'hi': 'hi',
    'ta': 'ta'
}

logging.basicConfig(level=logging.INFO)

@app.route('/greet', methods=['GET'])
def greet():
    # Check if the user has been greeted before
    if session.get('greeted'):
        # Redirect to /ask if the user has already been greeted
        return redirect(url_for('ask'))
    
    # Set the greeted flag in the session
    session['greeted'] = True
    
    # Generate the greeting message
    prompt = (
        "You are Cashie, an empathetic and helpful chatbot specifically designed to assist migrant workers in Singapore with queries related to DBS Bank. "
        "Don't mention migrant workers in response, use 'you' or any suitable words for replacement"
        "Tell them Cashie is capable of analsying scams by user uploaded screenshots of photos"
        "Your goal is to greet users in a friendly and welcoming manner while making them feel comfortable and at ease. Please generate a concise and warm, friendly greeting message for a user coming to Cashie for the first time."
        "Keep it 2-3 sentences"
    )
    greeting = get_groq_response(prompt)
    app.logger.info(f"Greeting response: {greeting}")
    return jsonify({"greeting": greeting}) 

def get_exchange_rate():
    url = "https://v6.exchangerate-api.com/v6/fec79fc704ea56a23e24186e/latest/SGD"
    try:
        response = requests.get(url)
        data = response.json()
        if data['result'] == 'success':
            exchange_rates = data['conversion_rates']
            return exchange_rates
        else:
            return "Error fetching exchange rates"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_pdf(pdf_path):
    try:
        text = ''
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

def chunk_text(text, chunk_size=500):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ''
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def find_relevant_chunk_via_similarity(chunks, query, top_n=1):
    vectorizer = TfidfVectorizer().fit_transform([query] + chunks)
    vectors = vectorizer.toarray()
    
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
    relevant_chunks_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    return [chunks[i] for i in relevant_chunks_indices if cosine_similarities[i] > 0.1]

def check_for_scam(image_path):
    img = Image.open(image_path)
    text = image_to_string(img)
    scam_keywords = [
    'scam', 'urgent', 'click here', 'won', 'free', 'money', 
    'prize', 'lottery', 'limited time', 'act now', 'exclusive offer', 
    'risk-free', 'congratulations', 'claim your prize', 'guaranteed', 
    'investment', 'cash', 'credit card', 'transfer', 
    'wire', 'verify', 'password', 'confidential', 'suspicious', 'update now', 
    'security alert', 'reward', 'refund', 'get rich', 'double your money', 
    'job offer', 'inheritance', 'urgent response needed', 'limited offer', 
    'claim now', 'pay now', 'your account', 'gift card', 'act immediately',
    'http:', 'https:', 'www.', '.com', '.net', '.xyz', '.ru', '.cn', '.info', 
    '.biz', '.top', '.cc', '.pw', '.live', '.site', '.online', '.tech', '.space', 
    'support', 'bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'buff.ly', 'shorte.st', 
    'login', 'verify', 'account', 'secure', 'update', 'support', 'paypal', 'bank', 
    'signin', 'password-reset', 'confirm', 'admin', 'alert', 'access',
    'immediate action required', 'investment opportunity', 'exclusive deal', 
    'easy money', 'no risk', 'limited time offer', 'private offer', 'tax refund', 
    'award', 'pre-approved', 'instant approval', 'unsecured loan', 'prepaid card', 
    'quick cash', 'guaranteed win', 'clearance sale', 'win big', 'crypto', 'bitcoin', 
    'bitcoins', 'blockchain', 'ICO', 'investment portfolio', 'financial freedom', 
    'wealth management', 'wealth transfer', 'free trial', 'deposit now', 
    'purchase now', 'easy approval', 'exclusive deal', 'pre-screened', 
    'fast-track', 'withdrawal request', 'fraudulent', 'money laundering', 
    'refund request', 'reimbursement', 'unverified', 'fake', 'phishing', 
    'identity theft', 'scammed', 'ransomware', 'malware', 'keylogger', 
    'fake invoice', 'payment request', 'unwanted subscription', 'fake job offer', 
    'credit repair', 'guaranteed loan', 'offshore investment', 'get paid', 
    'investment fraud', 'free gift', 'fake check', 'untraceable', 'questionable',
    'monthly installment',
]

    
    detected_keywords = [keyword for keyword in scam_keywords if keyword in text.lower()]

    if detected_keywords:
        explanation_prompt = (
            "The uploaded image contains suspicious content, likely indicating a scam. Keep it simple for people with basic knowledge of english "
            "Please explain why the following text from the image is considered a scam:\n\n"
            f"{text}\n\n"
        )
        
        scam_explanation = get_groq_response(explanation_prompt)
        
        return f"**Possible scam message detected!**\n\n{scam_explanation}"
    
    return "No scam detected. Your message appears safe, but always stay cautious when receiving unsolicited offers or requests for sensitive information."

def get_groq_response(prompt):
    try:
        query_content = (
            "You are Cashie, an empathetic and helpful chatbot designed to assist migrant workers in Singapore with DBS Bank queries. "
            "Don't mention migrant workers in response, use 'you' or any suitable words for replacement"
            "Focus on DBS Digibank app support, Singaporean banking services, and provide clear, direct answers specific to Singapore. "
            "Be concise, empathetic, and avoid lengthy explanations—use brief, reassuring responses that build confidence in financial decisions. "
            "Limit responses to essential information, with Singapore-specific context only, omitting services or regulations outside Singapore.\n\n"
            f"User Question: {prompt}\n\n"
            "Note: If relevant information is available, use only that content to answer the question concisely and directly. For queries involving banking, account setup, or scams, provide short, Singapore-specific steps and reassurance without extra detail."
        )

        # Send the request to Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query_content}],
            model="llama3-8b-8192",
        )

        print(f"Sending prompt to Groq: {query_content}")
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not session.get('greeted'):
        session['greeted'] = True
        greeting_prompt = (
            "You are Cashie, an empathetic and helpful chatbot specifically designed to assist migrant workers in Singapore with queries related to DBS Bank. "
            "Don't mention migrant workers in response, use 'you' or any suitable words for replacement"
            "Please generate a warm, friendly greeting message for a first-time user."
        )
        greeting = get_groq_response(greeting_prompt)
        return jsonify({"reply": greeting})
    
    incoming_msg = request.json.get('message').strip()

    if any(term in incoming_msg.lower() for term in ["remit rate", "remit", "rate"]):
        currency_code = extract_currency_from_message(incoming_msg)
        if currency_code:
            exchange_rates = get_exchange_rate()
            if currency_code in exchange_rates:
                rate = exchange_rates[currency_code]
                response_content = f"The current remit rate for {currency_code} is {rate} per 1 SGD."

                prompt = (
                "You are Cashie, an empathetic chatbot specifically designed to assist migrant workers in Singapore with queries related to DBS Bank. "
                "Don't mention migrant workers in response, use 'you' or any suitable words for replacement"
                "Calculate the amount that their family will receive as well"
                "Provide a friendly, conversational message to explain the following information concisely:\n\n"
                f"{response_content}"
            )
                response = get_groq_response(prompt)
            else:
                response = f"Sorry, I couldn't find the remit rate for {currency_code}."
        else:
            response = "Please specify the currency or country you're interested in."
        return jsonify({"reply": response})

    detected_language_code = detect(incoming_msg)
    print(f"Detected language code: {detected_language_code}")
    app_language_code = LANGDETECT_TO_APP_LANG_MAP.get(detected_language_code, 'en')
    incoming_msg_translated = (
        GoogleTranslator(source=app_language_code, target='en').translate(incoming_msg)
        if app_language_code != 'en' else incoming_msg
    )

    pdf_folder_path = 'uploads'
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    print(f"Attempting to read PDF file at: {pdf_files}")
    relevant_chunks = []

    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(pdf_folder_path, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_file_path)
            chunks = chunk_text(text)
            relevant_chunks += find_relevant_chunk_via_similarity(chunks, incoming_msg_translated, top_n=3)
        except ValueError as e:
            logging.info(f"Error processing PDF file {pdf_file}: {e}")

    if relevant_chunks:
        context = "\n".join(relevant_chunks)
        prompt = f"Based on this information: '{context}', answer the query: {incoming_msg_translated}"
    else:
        prompt = incoming_msg_translated

    response = get_groq_response(prompt)

    final_reply = response if len(response.split('.')) < 100 else '. '.join(response.split('.')[:100])

    if app_language_code != 'en':
        final_reply = GoogleTranslator(source='en', target=app_language_code).translate(final_reply)
    logging.info(f"Final Reply {final_reply}")

    return jsonify({"reply": final_reply})



@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']
    if not file.filename.endswith(('png', 'jpg', 'jpeg','PNG')):
        return "File type not allowed", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    scam_check_result = check_for_scam(file_path)
    os.remove(file_path)

    return scam_check_result

def extract_currency_from_message(message):
    currencies = {
    'INR': ['inr', 'india', 'indian rupee'],
    'PHP': ['php', 'philippines', 'peso'],
    'MYR': ['myr', 'malaysia', 'ringgit', 'rm', 'RM'],
    'IDR': ['idr', 'indonesia', 'rupiah'],
    'BDT': ['bdt', 'bangladesh', 'taka'],
    'CNY': ['cny', 'china', 'chinese yuan', 'yuan', 'renminbi', 'rmb'],
    'VND': ['vnd', 'vietnam', 'dong', 'vietnamese dong'],
    'THB': ['thb', 'thailand', 'baht', 'thai baht'],
    'MMK': ['mmk', 'myanmar', 'kyat', 'burmese kyat'],
    'KHR': ['khr', 'cambodia', 'riel', 'cambodian riel']
}


    for currency_code, terms in currencies.items():
        for term in terms:
            if term in message.lower():
                return currency_code
    return None

if __name__ == '__main__':
    app.run(debug=True)
