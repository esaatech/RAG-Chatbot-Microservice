# Usage Examples

## Python Client
python
import requests
BASE_URL = "http://localhost:8090"
def upload_document(file_path, config=None):
files = {'file': open(file_path, 'rb')}
data = {'prompt_config': config} if config else None
response = requests.post(f"{BASE_URL}/documents/upload", files=files, json=data)
return response.json()
def query_document(key, query):
response = requests.post(
f"{BASE_URL}/documents/{key}/query",
params={'query': query}
)
return response.json()
def update_config(key, new_config):
response = requests.put(f"{BASE_URL}/documents/{key}/update-config",
json={'prompt_config': new_config}
)
return response.json()
Example usage
doc_key = upload_document('sample.pdf')['key']
answer = query_document(doc_key, "What is this document about?")
print(answer)