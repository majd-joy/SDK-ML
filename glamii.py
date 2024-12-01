import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

text = speech_to_text()


import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def get_response(text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=50
    )
    return response.choices[0].text.strip()

response_text = get_response(text)
print("AI Response: " + response_text)
from gtts import gTTS
import os

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

text_to_speech(response_text)

