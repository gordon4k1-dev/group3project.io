from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model directly from the models/ directory if a .gguf file exists
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = None

# Search for a .gguf model in the models/ directory
if os.path.exists(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".gguf"):
            MODEL_PATH = os.path.join(MODEL_DIR, f)
            break

llm = None
if Llama is not None and MODEL_PATH is not None:
    try:
        # Load the Phi-3 Mini GGUF model
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get("/search")
def search(prompt: str):
    if not prompt:
        return {"response": "Prompt is required."}

    if llm is not None:
        try:
            # Generate response using llama-cpp-python
            output = llm(
                f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
                max_tokens=256,
                stop=["<|end|>"]
            )
            ai_text = output["choices"][0]["text"].strip()
            return {"response": ai_text}
        except Exception as e:
            return {"response": f"Error generating response: {str(e)}"}
    else:
        # Fallback error message for local testing when model isn't downloaded yet
        err_msg = "Model not found or llama-cpp-python not installed. "
        if MODEL_PATH is None:
            err_msg += f"Please place your .gguf model file inside the '{os.path.abspath(MODEL_DIR)}' directory."
        return {"response": err_msg}

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=True)
