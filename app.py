import streamlit as st
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, BaseModel
import requests
import json

class FastAPILLM(LLM, BaseModel):
    endpoint_url: str = Field(description="URL of the FastAPI endpoint")
    model_name: str = Field(default="llama3.1:8b", description="Name of the model")
    
    class Config:
        arbitrary_types_allowed = True

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> str:
        return "".join(chunk.text for chunk in self._stream(prompt, stop, run_manager, **kwargs))

    def _stream(
        self, prompt: str, stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Iterator[GenerationChunk]:
        try:
            with requests.post(
                self.endpoint_url,
                json={"prompt": prompt},
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            generated_text = data.get("response", "")
                            if generated_text:  # Only yield if there's actual content
                                yield GenerationChunk(text=generated_text)
                        except json.JSONDecodeError:
                            continue
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
            full_text = ""
            # Create a spinner while generating
            with st.spinner("Generating..."):
                for chunk in llm._stream(prompt):
                    if chunk.text:
                        full_text += chunk.text
                        # Update the output in real-time
                        output_placeholder.markdown(f"**Generated Text:**\n\n{full_text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a prompt to continue.")
