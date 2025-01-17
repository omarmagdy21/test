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
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make a call to the FastAPI endpoint."""
        try:
            # Make the request to your FastAPI endpoint
            response = requests.post(
                self.endpoint_url,
                params={"prompt": prompt},
                timeout=60
            )
            response.raise_for_status()
            
            # Parse the JSON response from Ollama
            response_data = response.json()
            
            # Extract the generated text
            if isinstance(response_data, dict) and "response" in response_data:
                generated_text = response_data["response"]
            else:
                generated_text = response_data
            
            # Handle stop words if provided
            if stop:
                for stop_sequence in stop:
                    if stop_sequence in generated_text:
                        generated_text = generated_text[:generated_text.index(stop_sequence)]
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling FastAPI endpoint: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing response from FastAPI endpoint: {str(e)}")

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream tokens from the FastAPI endpoint."""
        response = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        yield GenerationChunk(text=response)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "endpoint_url": self.endpoint_url
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "fastapi_ollama"

llm = FastAPILLM(
                endpoint_url="http://34.57.134.25:8000/generate",
                model_name="llama3.1:8b",
            )
response = llm.invoke("tell me information about salah")