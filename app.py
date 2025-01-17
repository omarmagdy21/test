import streamlit as st
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, BaseModel
import requests
import json

class FastAPILLM(LLM, BaseModel):
    """Custom LLM that uses a FastAPI endpoint running Ollama."""
    
    endpoint_url: str = Field(description="URL of the FastAPI endpoint")
    model_name: str = Field(default="llama3.1:8b", description="Name of the model")
    
    class Config:
        arbitrary_types_allowed = True

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        """Fallback non-streaming call."""
        return self._stream(prompt, stop, run_manager, **kwargs).__next__().text

    def _stream(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[GenerationChunk]:
        """Stream tokens from the FastAPI endpoint."""
        try:
            with requests.post(self.endpoint_url, params={"prompt": prompt}, stream=True, timeout=60) as response:
                response.raise_for_status()
                buffer = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            data = json.loads(decoded_line)
                            generated_text = data.get("response", "")
                            yield GenerationChunk(text=generated_text)
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
        except requests.exceptions.RequestException as e:
            yield GenerationChunk(text=f"Error: {str(e)}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "endpoint_url": self.endpoint_url}

    @property
    def _llm_type(self) -> str:
        return "fastapi_ollama"

# Initialize FastAPILLM
llm = FastAPILLM(endpoint_url="http://34.57.134.25:8000/generate", model_name="llama3.1:8b")

# Streamlit App
st.title("FastAPILLM Streaming Demo")

# Input for user prompt
prompt = st.text_input("Enter your prompt:")

# Button to invoke LLM
if st.button("Generate"):
    if prompt:
        output_placeholder = st.empty()
        try:
            response_stream = llm._stream(prompt)
            full_text = ""
            for chunk in response_stream:
                full_text += chunk.text
                output_placeholder.markdown(f"**Generated Text:**\n\n{full_text}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt to continue.")
