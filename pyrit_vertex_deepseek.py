#!/usr/bin/env python3
"""
PyRIT Vertex AI Integration - DeepSeek Publisher Model
Connects to DeepSeek v3.1 via Publisher Model API (Model Garden)
"""

import os
from typing import Optional
import json

class VertexDeepSeekTarget:
    """
    PyRIT-compatible chat target for Vertex AI DeepSeek Publisher Model
    Uses the generateContent API, not endpoint.predict()
    """
    
    def __init__(self, 
                 model_id: str = "gen-lang-client-0001839742",
                 project_id: Optional[str] = None, 
                 region: Optional[str] = None):
        """
        Initialize Vertex AI DeepSeek Publisher Model target
        
        Args:
            model_id: Publisher model ID (default: gen-lang-client-0001839742 for DeepSeek v3.1)
            project_id: GCP project ID (defaults to environment or ADC)
            region: GCP region (defaults to us-west2)
        """
        try:
            from google.cloud import aiplatform
            from google.auth import default
            import google.auth.transport.requests
            
            # Get credentials and project
            credentials, default_project = default()
            
            self.model_id = model_id
            self.project_id = project_id or default_project or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.region = region or os.getenv("GCP_REGION", "us-west2")
            
            if not self.project_id:
                raise ValueError(
                    "Project ID required. Set GOOGLE_CLOUD_PROJECT environment variable "
                    "or pass project_id parameter"
                )
            
            # Store credentials for API calls
            self.credentials = credentials
            self.request = google.auth.transport.requests.Request()
            self.credentials.refresh(self.request)
            
            # Build the API endpoint URL
            self.api_endpoint = (
                f"https://{self.region}-aiplatform.googleapis.com/v1/"
                f"projects/{self.project_id}/locations/{self.region}/"
                f"publishers/google/models/{self.model_id}:generateContent"
            )
            
        except ImportError as e:
            raise ImportError(
                "google-cloud-aiplatform not installed. "
                "Install with: pip install google-cloud-aiplatform"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vertex AI Publisher Model: {e}") from e
    
    def send_prompt(self, prompt_text: str, **kwargs):
        """
        Send a prompt to the DeepSeek Publisher Model using generateContent API
        
        Args:
            prompt_text: The prompt to send
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Response text from the model
        """
        try:
            import requests
            
            # Prepare the request payload
            # Format for Vertex AI Publisher Model generateContent API
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt_text
                            }
                        ]
                    }
                ]
            }
            
            # Add generation config if parameters provided
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                generation_config["maxOutputTokens"] = kwargs["max_tokens"]
            if "maxOutputTokens" in kwargs:
                generation_config["maxOutputTokens"] = kwargs["maxOutputTokens"]
            if "top_p" in kwargs:
                generation_config["topP"] = kwargs["top_p"]
            if "top_k" in kwargs:
                generation_config["topK"] = kwargs["top_k"]
            
            # Only add generationConfig if we have parameters
            if generation_config:
                payload["generationConfig"] = generation_config
            
            # Get access token
            self.credentials.refresh(self.request)
            access_token = self.credentials.token
            
            # Make the API call
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text from the response
            # Response structure: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            
            # Fallback: return raw response if structure is different
            return json.dumps(result, indent=2)
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - {e.response.text[:200]}"
            return f"Error calling Vertex AI Publisher Model: {error_msg}"
        except Exception as e:
            return f"Error calling Vertex AI Publisher Model: {e}"
    
    def test_connection(self):
        """Test the connection to the Publisher Model"""
        try:
            test_response = self.send_prompt("Say hello in one word")
            if test_response and not test_response.startswith("Error"):
                return True, test_response
            else:
                return False, test_response
        except Exception as e:
            return False, str(e)


