import streamlit as st
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth


bedrock_runtime_client = boto3.client('bedrock-runtime',region_name='us-east-1')
s3_client = boto3.client('s3')

session = boto3.Session()
host = '<input your endpoint>'
region = '<input your region>'
service = 'aoss'
index_name = "<input your index name>"


def create_opensearch_client(session, host, region, service):
    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)

    os_client = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        pool_maxsize = 20
    )
    return os_client

def invoke_embedding_model(input):
    response = bedrock_runtime_client.invoke_model(
        body=json.dumps({
            'texts': input,
            'input_type': 'search_document'
        }),
        modelId = 'cohere.embed-multilingual-v3',
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("embeddings")

def generate_presigned_url(bucket_name, object_key, expiration=86400):
    url = s3_client.generate_presigned_url(
        ClientMethod = 'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=expiration
    )
    return url


def invoke_llm_model(input):
    
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": input,
            }
        ],
    }
    messages = [user_message]
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "top_k": 250,
        }
    )
    
    modelId="anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"
    
    response = bedrock_runtime_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read().decode())
    return response_body.get("content")[0]["text"]

def answer_query(user_input, index_name, os_client):
    print(user_input)
    userQuery = user_input
    userVectors = invoke_embedding_model([userQuery])[0]
    query = {
        "size": 2,
        "query": {
            "knn": {
                "processed_element_embedding": {
                    "vector": userVectors, 
                    "k": 2
                }
            }
        },
    }
    response = os_client.search(
        body=query,
        index=index_name
    )
    
    hits = response['hits']['hits']
    prompt_template = """
あなたはユーザの質問に答えるAIアシスタントです。<質問></質問>の質問に対して、<参考ドキュメント></参考ドキュメント>の内容に基づき、<回答のルール></回答のルール>に従って回答を行なってください。
<質問>
{question}
</質問>
<参考ドキュメント>
{context}
</参考ドキュメント>
<回答のルール>
* 必ず<参考ドキュメント></参考ドキュメント>をもとに回答してください。
* 回答文以外の文字列、および「参考ドキュメント」「画像」「表」といった言葉は一切出力しないでください。挨拶は不要です。
</回答のルール>
"""
    context = []
    attachment = {}
    for hit in hits:
        context.append(hit['_source']['processed_element'])
        if not attachment and hit['_source']['raw_element_type'] == 'image':
            attachment['bucket'] = hit['_source']['s3_bucket']
            attachment['image_s3_path'] = hit['_source']['image_s3_path']
    print(context)

    llm_prompt = prompt_template.format(context='\n'.join(context),question=user_input)
    output = invoke_llm_model(llm_prompt)
    
    return output, attachment

def main():
    
    st.set_page_config(page_title="🤖 Multimodal RAG demo with Bedrock", layout="wide")
    st.title("🤖 Multimodal RAG demo with Bedrock")
    
    os_client = create_opensearch_client(session, host, region, service)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_msg = st.chat_input("ここにメッセージを入力")
    
    if user_msg:
        
        with st.chat_message("user"):
            st.markdown(user_msg)
        # append the question and the role (user) as a message to the session state
        st.session_state.messages.append({"role": "user",
                                          "content": user_msg})
        # respond as the assistant with the answer
        with st.chat_message("assistant"):
            # putting a spinning icon to show that the query is in progress
            with st.status("Determining the best possible answer!", expanded=True) as status:
                # making sure there are no messages present when generating the answer
                message_placeholder = st.empty()
                # passing the question into the OpenSearch search function, which later invokes the llm
                answer, attachment = answer_query(user_msg, index_name, os_client)
                # writing the answer to the front end
                message_placeholder.markdown(f"{answer}")
                
                if attachment:
                    message_placeholder = st.empty()
                    message_placeholder.image(generate_presigned_url(attachment['bucket'], attachment['image_s3_path']))
                # showing a completion message to the front end
                # status.update(label="Question Answered...", state="complete", expanded=False)
        # appending the results to the session state
        st.session_state.messages.append({"role": "assistant","content": answer})

if __name__ == '__main__':
    main()