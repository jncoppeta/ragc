# How to Start
### 1. Clone repository
- `git clone https://github.com/jncoppeta/ragc.git`
### 2. Add your PDFs to the ./pdf folder
### 3. Run the startup script
- `bash start.sh`
- The script waits until all the containers are healthly are it will output this message on completion:
```
==================================================
All containers have started successfully!
You can access the API documentation at: http://localhost:8000/docs
==================================================
```

# Usage Guide
### UI
- Go to `http://localhost:8000/docs` after successful startup to acess the SwaggerUI docs page
- Click on the `POST /quesiton` box to open the dropdown
- Click `Try it out` in the top right to edit the request JSON
- Enter your question as the value to the `"question"` key and click `Execute`
### API
- python
```
import requests

url = "http://localhost:8000/question"
data = {
    "question": "What is the capital of France?"  # Replace with the actual question
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Request was successful!")
    print(response.json())  # Print the response body (if any)
else:
    print(f"Request failed with status code {response.status_code}")
```
- bash
```
#!/bin/bash

URL="http://localhost:8000/question"
QUESTION="What is the capital of France?"  # Replace with the actual question

curl -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$QUESTION\"}"
```
