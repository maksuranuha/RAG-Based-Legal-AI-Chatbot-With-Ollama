## Basic Concept for Ollama 
What is Ollama? 
- Free, open-source tool
- Build offline, Manage & Run LLM Models
- Privacy-focused projects
- Provides HTTP Server
- Call model from any app
- can use Ollama to experiment with AI without needing powerful cloud servers.

Basis for choosing this :  
- Suppose, this legal law AI is for a local law firm so  I want it to run offline to protect sensitive data and ensure it works even without internet access.

Setting Up Ollama : 
- Download Ollama from ollama.com & Install it. 
- Go to Ollama Librray and choose preferred model acording to your PC HW
  Go to CMD and run : ollama run chosen_model_name, example:  
  > ollama run mistral
- If you download multiple models, just check with Ollama_list

Change model names in File : 
 - vector_database.py -> model_name = "chosen_model_name" (eg. model_name = "mistral:latest")
 - main.py -> model_name = "chosen_model_name" (eg. model_name = "mistral:latest")

Start With  - ollama serve 
and           ollama run chosen_model_name