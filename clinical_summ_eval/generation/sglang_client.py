"""
SGLang Client with Server Management

Provides a wrapper around SGLang's OpenAI-compatible API with utilities for:
- Starting/stopping SGLang servers
- Managing conversations with memory
- Tracking API usage
"""

import subprocess
import time
import requests
from typing import Dict, List, Optional, Any
from openai import OpenAI


class SGLangServer:
    """Manages SGLang server lifecycle."""
    
    def __init__(
        self,
        model_path: str,
        port: int = 30000,
        host: str = "0.0.0.0",
        tensor_parallel_size: int = 1,
        mem_fraction_static: float = 0.8,
        max_running_requests: Optional[int] = None,
        additional_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SGLang server configuration.
        
        Args:
            model_path: HuggingFace model path or local path
            port: Server port (default: 30000)
            host: Server host (default: 127.0.0.1)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            mem_fraction_static: Fraction of GPU memory for static allocation
            max_running_requests: Maximum concurrent requests
            additional_args: Additional server arguments as dict
                Example: {"context_length": 8192, "disable_cuda_graph": True}
        """
        self.model_path = model_path
        self.port = port
        self.host = host
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.max_running_requests = max_running_requests
        self.additional_args = additional_args or {}
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}/v1"
    
    def start(self, wait_time: int = 30) -> bool:
        """
        Start the SGLang server.
        
        Args:
            wait_time: Seconds to wait for server to be ready
            
        Returns:
            True if server started successfully
        """
        if self.is_running():
            print(f"Server already running at {self.base_url}")
            return True
        
        # Build command
        cmd = [
            "uv","run","python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--port", str(self.port),
            "--host", self.host
        ]
        
        if self.max_running_requests:
            cmd.extend(["--max-running-requests", str(self.max_running_requests)])
        
        # Add additional arguments
        for key, value in self.additional_args.items():
            arg_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(arg_name)
            else:
                cmd.extend([arg_name, str(value)])
        
        print(f"Starting SGLang server: {' '.join(cmd)}")
        
        # Start server process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        print(f"Waiting {wait_time}s for server to start...")
        time.sleep(wait_time)
        
        if self.is_running():
            print(f"✓ Server running at {self.base_url}")
            return True
        else:
            print("✗ Server failed to start")
            if self.process:
                print("Server output:")
                stdout, stderr = self.process.communicate(timeout=120)
                print(stdout)
                print(stderr)
            return False
    
    def stop(self):
        """Stop the SGLang server."""
        if self.process:
            print("Stopping SGLang server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("✓ Server stopped")
            except subprocess.TimeoutExpired:
                print("Server didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
                print("✓ Server killed")
            self.process = None
        else:
            print("No server process to stop")
    
    def is_running(self) -> bool:
        """Check if server is running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class SGLangClient:
    """Client for interacting with SGLang server via OpenAI-compatible API."""
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:30000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.1
    ):
        """
        Initialize SGLang client.
        
        Args:
            model: Model name (must match what's loaded on the server)
            base_url: SGLang server URL
            api_key: API key (use "EMPTY" for local servers)
            temperature: Sampling temperature
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
    
    def generate(
            self,
            prompt: str,
            system_prompt: str = "",
            use_memory: bool = False,
            response_format: Optional[Dict] = None,
            temperature: Optional[float] = None,
        ) -> Dict[str, Any]:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            use_memory: Whether to maintain conversation history
            response_format: JSON schema for structured output
            temperature: Override default temperature

        Returns:
            Dict with 'reply', 'reasoning' (str or None), and 'usage' keys.
            'reasoning' is None when the model does not produce reasoning content
            (either because reasoning is not enabled on the server, or the model
            does not support it).
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if using memory
        if use_memory and self.conversation_history:
            messages.extend(self.conversation_history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Build request kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        # Add JSON response format if requested
        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        # Make API call
        response = self.client.chat.completions.create(**kwargs, max_tokens=32_000, extra_body={"chat_template_kwargs": {"enable_thinking": False}})

        message = response.choices[0].message
        reply = message.content

        # Robust reasoning extraction:
        # - Some SDK versions don't expose `reasoning_content` at all -> AttributeError-safe via getattr
        # - Some return the attribute but as None or "" -> normalize to None
        # - Strip whitespace so trivial whitespace-only reasoning is treated as None
        reason = getattr(message, "reasoning_content", None)
        if isinstance(reason, str):
            reason = reason.strip() or None
        elif reason is not None and not isinstance(reason, str):
            # Defensive: if some SDK returns a non-string (e.g. list of blocks), coerce
            try:
                reason = str(reason).strip() or None
            except Exception:
                reason = None

        # Update conversation history if using memory
        if use_memory:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": reply})

        # Update usage statistics
        usage = response.usage
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.call_count += 1

        return {
            "reply": reply,
            "reasoning": reason,  # None if model produced no reasoning
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get cumulative usage statistics."""
        return {
            "total_calls": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_call": self.total_tokens / self.call_count if self.call_count > 0 else 0,
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0


# ============================================================
# Convenience Functions
# ============================================================

def create_server_and_client(
    model_path: str,
    port: int = 30000,
    temperature: float = 0.1,
    server_kwargs: Optional[Dict] = None,
    client_kwargs: Optional[Dict] = None,
    auto_start: bool = True
) -> tuple[SGLangServer, SGLangClient]:
    """
    Convenience function to create and optionally start server + client.
    
    Args:
        model_path: HuggingFace model path
        port: Server port
        temperature: Client temperature
        server_kwargs: Additional server configuration
        client_kwargs: Additional client configuration
        auto_start: Whether to start the server immediately
        
    Returns:
        (server, client) tuple
    """
    server_kwargs = server_kwargs or {}
    client_kwargs = client_kwargs or {}
    
    server = SGLangServer(model_path=model_path, port=port, **server_kwargs)
    
    if auto_start:
        server.start()
    
    # Extract model name from path for client
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    
    client = SGLangClient(
        model=model_name,
        base_url=f"http://127.0.0.1:{port}/v1",
        temperature=temperature,
        **client_kwargs
    )
    
    return server, client