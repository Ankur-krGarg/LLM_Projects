# Voice Assistant <br>
This code implements a voice assistant using Streamlit, Langchain, and Cohere. It provides a user interface for interacting with the assistant, transcribing audio, and retrieving information from a knowledge base.

**Key Features**<br>

Audio Transcription: Records audio using Streamlit's audio recorder and transcribes it using the Cohere Whisper API.<br>
Knowledge Base: Uses Langchain's DeepLake to store and retrieve embeddings from a set of documentation.<br>
Chat Interface: Provides a Streamlit-based chat interface for interaction with the assistant.<br>
Text-to-Speech: Converts the assistant's responses to audio using Pyttsx3.<br>


**Running the Code**<br>

Set up Environment:
Install required Python packages: pip install -r requirements.txt<br>
Set API Keys:
Create a .env file in the project directory and set the following environment variables:<br>
COHERE_API_KEY: Your Cohere API Key<br>
Create a Dataset:
Create a DeepLake dataset for storing embeddings, following the instructions in the code (lines 24-26).<br>
Run the Script:
Execute the code using streamlit run voice_assistant.py<br>
Interact with the Assistant:
Use the chat interface or record audio to interact with the assistant.<br>

**Code Breakdown**<br>
Load Documents: The code loads and splits documentation files to create chunks for embedding.<br>
Embed and Store: CohereEmbeddings are used to create embeddings for the document chunks, which are then stored in a DeepLake database.<br>
Search and Retrieve: The assistant searches the DeepLake database based on user input and retrieves the most relevant information.<br>
Chat Interface: The display_conversation function handles the display of chat messages and audio playback.
Text-to-Speech: The text_to_speech_pyttsx3 function converts the assistant's responses to speech using Pyttsx3.
