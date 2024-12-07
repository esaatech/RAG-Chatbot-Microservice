# API Documentation

## Endpoints

### Document Management

#### Upload Document
`POST /documents/upload`

Upload and process a new document.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - file: Document file (PDF, TXT, DOCX, CSV)
  - prompt_config (optional): JSON configuration
json
{
"prompt_config": {
"company_name": "YourCompany",
"agent_role": "support agent",
"response_style": "concise",
"tone": "professional"
}
}
**Response:**
json
{
"key": "uuid-string",
"message": "Document processed successfully",
"config": {
"company_name": "YourCompany",
"agent_role": "support agent",
"response_style": "concise",
"tone": "professional"
}
}

#### Query Document
`POST /documents/{key}/query`

Query a processed document.

**Request:**
- Method: POST
- Query Parameters:
  - query: string (The question to ask)

**Response:**
json
{
"answer": "Response based on document content"
}

Update Document
PUT /documents/{document_key}/update
Update an existing document with new content and/or configuration.
Request:
Method: PUT
Content-Type: multipart/form-data
Parameters:
document_key: string (path parameter)
Body:
file: Document file (PDF, TXT, DOCX, CSV)
prompt_config (optional): JSON string configuration
Example prompt_config:

{
    "company_name": "UpdatedCo",
    "agent_role": "specialist",
    "response_style": "detailed",
    "tone": "professional"
}
Response:

{
    "old_key": "previous-uuid",
    "new_key": "new-uuid",
    "prompt_config": {
        "company_name": "UpdatedCo",
        "agent_role": "specialist",
        "response_style": "detailed",
        "tone": "professional"
    },
    "message": "Document updated successfully"
}
Error Responses:
400: Invalid JSON in prompt_config
404: Document not found
500: Server error



#### Update Configuration
`PUT /documents/{key}/update-config`

Update document configuration.

**Request:**
- Method: PUT
- Content-Type: application/json
- Body:
json
{
"prompt_config": {
"company_name": "NewCompany",
"tone": "friendly",
"response_style": "detailed",
"agent_role": "expert"
}
}

#### Delete Document
`DELETE /documents/{key}`

Delete a document and its associated data.

#### Get Cache Stats
`GET /documents/cache-stats`

Get cache statistics.

## Error Handling
All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Server Error