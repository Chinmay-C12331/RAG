# install required module as we are using gemini api here we are using -->  !pip install -U langchain-google-genai
from langchain import messages
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
#i have used googlee colab so my secret key is in colab!!!!
api_key=userdata.get('GEMINI_API_KEY')
system_msg=SystemMessage("You are a helpful assistant.")
human_msg=HumanMessage("what is rag?")
messages=[system_msg,human_msg]
model=init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=api_key
)
response = model.invoke(messages)
print(response.content)
