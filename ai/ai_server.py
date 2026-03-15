from fastapi import FastAPI
import requests
import uvicorn

app = FastAPI()

@app.get("/search")
def search(prompt: str):
    # Endpoint for a standard local AI runtime like Ollama
    api_url = "http://localhost:11434/api/generate"

    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        ai_text = data.get("response", "")
        return {"response": ai_text}
    except Exception as e:
        # If the local AI isn't running or there is an error, return a graceful error message in the response
        return {"response": f"Error generating response from local AI: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=True)
