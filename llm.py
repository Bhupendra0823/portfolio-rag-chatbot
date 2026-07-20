from dotenv import load_dotenv
load_dotenv()
import os
from langchain_aws import ChatBedrock, BedrockEmbeddings


def get_aws_embeddings():
    return BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL"),
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),   
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
def get_aws_llm():
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_LLM_MODEL"),
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  
    )

    
 