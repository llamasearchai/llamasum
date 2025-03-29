"""REST API server for LlamaSum using FastAPI."""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware

from llamasum.summarizer import LlamaSummarizer
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.multidoc import MultiDocSummarizer
from llamasum.config import (
    SummarizerConfig, 
    HierarchicalConfig, 
    MultiDocConfig,
    load_config_from_env
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for summarizers
basic_summarizer = None
hierarchical_summarizer = None
multidoc_summarizer = None

# Define API models
class SummarizeRequest(BaseModel):
    """Request model for basic summarization."""
    text: str = Field(..., description="Text to summarize")
    min_length: Optional[int] = Field(None, description="Minimum length of summary")
    max_length: Optional[int] = Field(None, description="Maximum length of summary")
    ratio: Optional[float] = Field(
        None, 
        description="Compression ratio for extractive phase (0.0-1.0)"
    )
    model_name: Optional[str] = Field(
        None, 
        description="Model to use (overrides default config)"
    )
    device: Optional[str] = Field(
        None, 
        description="Device to use (cpu, cuda, mps)"
    )


class HierarchicalRequest(SummarizeRequest):
    """Request model for hierarchical summarization."""
    levels: Optional[int] = Field(
        None, 
        description="Number of summary levels to generate"
    )
    include_ultra_short: Optional[bool] = Field(
        None, 
        description="Include ultra-short summary"
    )
    method: Optional[str] = Field(
        "standard", 
        description="Summarization method (standard, sections, clustering)"
    )


class MultiDocRequest(BaseModel):
    """Request model for multi-document summarization."""
    documents: List[str] = Field(..., description="List of documents to summarize")
    min_length: Optional[int] = Field(None, description="Minimum length of summary")
    max_length: Optional[int] = Field(None, description="Maximum length of summary")
    strategy: Optional[str] = Field(
        None, 
        description="Strategy for multi-document summarization"
    )
    model_name: Optional[str] = Field(
        None, 
        description="Model to use (overrides default config)"
    )
    device: Optional[str] = Field(
        None, 
        description="Device to use (cpu, cuda, mps)"
    )
    queries: Optional[List[str]] = Field(
        None, 
        description="List of queries for query-focused summarization"
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="LlamaSum API",
        description="API for advanced text summarization",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize summarizers at startup
    @app.on_event("startup")
    async def startup_event():
        global basic_summarizer, hierarchical_summarizer, multidoc_summarizer
        
        try:
            # Try to load config from environment variables
            env_config = load_config_from_env()
            
            # Create default configurations with env values if available
            basic_config = SummarizerConfig(**env_config)
            hier_config = HierarchicalConfig(**env_config)
            multidoc_config = MultiDocConfig(**env_config)
            
            # Initialize summarizers
            logger.info(f"Initializing summarizers with model: {basic_config.model_name}")
            basic_summarizer = LlamaSummarizer(basic_config)
            hierarchical_summarizer = HierarchicalSummarizer(hier_config)
            multidoc_summarizer = MultiDocSummarizer(multidoc_config)
            
            logger.info("All summarizers initialized successfully")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            logger.exception("Initialization error")
    
    # Define API endpoints
    @app.get("/")
    async def root():
        """Root endpoint returning API info."""
        return {
            "name": "LlamaSum API",
            "version": "0.1.0",
            "description": "API for advanced text summarization",
            "endpoints": [
                "/summarize",
                "/summarize/hierarchical",
                "/summarize/multidoc"
            ],
            "status": "active"
        }
    
    @app.post("/summarize")
    async def summarize(request: SummarizeRequest):
        """Basic summarization endpoint.
        
        Args:
            request: Summarization request
            
        Returns:
            Dictionary with summary and metadata
        """
        global basic_summarizer
        
        if not basic_summarizer:
            raise HTTPException(
                status_code=500, 
                detail="Summarizer not initialized"
            )
        
        try:
            # Create a temporary config with request parameters
            config_kwargs = {}
            if request.model_name:
                config_kwargs["model_name"] = request.model_name
            if request.device:
                config_kwargs["device"] = request.device
            if request.min_length:
                config_kwargs["min_length"] = request.min_length
            if request.max_length:
                config_kwargs["max_length"] = request.max_length
            if request.ratio:
                config_kwargs["extractive_ratio"] = request.ratio
            
            # Use default summarizer if no custom settings
            if not config_kwargs:
                summary = basic_summarizer.summarize(
                    request.text,
                    min_length=request.min_length,
                    max_length=request.max_length,
                    ratio=request.ratio
                )
            else:
                # Create a temporary summarizer with custom config
                temp_config = SummarizerConfig(**config_kwargs)
                temp_summarizer = LlamaSummarizer(temp_config)
                summary = temp_summarizer.summarize(
                    request.text,
                    min_length=request.min_length,
                    max_length=request.max_length,
                    ratio=request.ratio
                )
            
            # Prepare response
            return {
                "summary": summary,
                "input_length": len(request.text),
                "summary_length": len(summary),
                "input_tokens": len(request.text.split()),
                "summary_tokens": len(summary.split()),
                "compression_ratio": round(
                    len(summary.split()) / len(request.text.split()), 2
                ) if len(request.text.split()) > 0 else 0
            }
            
        except Exception as e:
            logger.exception("Error during summarization")
            raise HTTPException(
                status_code=500,
                detail=f"Summarization error: {str(e)}"
            )
    
    @app.post("/summarize/hierarchical")
    async def summarize_hierarchical(request: HierarchicalRequest):
        """Hierarchical summarization endpoint.
        
        Args:
            request: Hierarchical summarization request
            
        Returns:
            Dictionary with hierarchical summary and metadata
        """
        global hierarchical_summarizer
        
        if not hierarchical_summarizer:
            raise HTTPException(
                status_code=500, 
                detail="Hierarchical summarizer not initialized"
            )
        
        try:
            # Create a temporary config with request parameters
            config_kwargs = {}
            if request.model_name:
                config_kwargs["model_name"] = request.model_name
            if request.device:
                config_kwargs["device"] = request.device
            if request.min_length:
                config_kwargs["min_length"] = request.min_length
            if request.max_length:
                config_kwargs["max_length"] = request.max_length
            if request.levels:
                config_kwargs["levels"] = request.levels
            if request.include_ultra_short is not None:
                config_kwargs["include_ultra_short"] = request.include_ultra_short
            
            # Use default summarizer if no custom settings
            if not config_kwargs:
                summarizer = hierarchical_summarizer
            else:
                # Create a temporary summarizer with custom config
                temp_config = HierarchicalConfig(**config_kwargs)
                summarizer = HierarchicalSummarizer(temp_config)
            
            # Choose method based on request
            if request.method == "sections":
                result = summarizer.summarize_by_sections(request.text)
            elif request.method == "clustering":
                result = summarizer.summarize_by_clustering(request.text)
            else:
                result = summarizer.summarize(request.text)
            
            # Add metadata to result
            result["input_length"] = len(request.text)
            result["input_tokens"] = len(request.text.split())
            
            return result
            
        except Exception as e:
            logger.exception("Error during hierarchical summarization")
            raise HTTPException(
                status_code=500,
                detail=f"Hierarchical summarization error: {str(e)}"
            )
    
    @app.post("/summarize/multidoc")
    async def summarize_multidoc(request: MultiDocRequest):
        """Multi-document summarization endpoint.
        
        Args:
            request: Multi-document summarization request
            
        Returns:
            Dictionary with multi-document summary and metadata
        """
        global multidoc_summarizer
        
        if not multidoc_summarizer:
            raise HTTPException(
                status_code=500, 
                detail="Multi-document summarizer not initialized"
            )
        
        try:
            # Create a temporary config with request parameters
            config_kwargs = {}
            if request.model_name:
                config_kwargs["model_name"] = request.model_name
            if request.device:
                config_kwargs["device"] = request.device
            if request.min_length:
                config_kwargs["min_length"] = request.min_length
            if request.max_length:
                config_kwargs["max_length"] = request.max_length
            if request.strategy:
                config_kwargs["strategy"] = request.strategy
            
            # Use default summarizer if no custom settings
            if not config_kwargs:
                summarizer = multidoc_summarizer
            else:
                # Create a temporary summarizer with custom config
                temp_config = MultiDocConfig(**config_kwargs)
                summarizer = MultiDocSummarizer(temp_config)
            
            # Choose between regular and query-focused summarization
            if request.queries:
                result = summarizer.summarize_with_queries(
                    request.documents,
                    request.queries,
                    max_length=request.max_length
                )
            else:
                result = summarizer.summarize(
                    request.documents,
                    max_length=request.max_length
                )
            
            # Add metadata to result
            total_input_length = sum(len(doc) for doc in request.documents)
            total_input_tokens = sum(len(doc.split()) for doc in request.documents)
            
            result["input_length"] = total_input_length
            result["input_tokens"] = total_input_tokens
            result["num_documents"] = len(request.documents)
            
            return result
            
        except Exception as e:
            logger.exception("Error during multi-document summarization")
            raise HTTPException(
                status_code=500,
                detail=f"Multi-document summarization error: {str(e)}"
            )
    
    return app


app = create_app()


def run_server(
    host: str = "0.0.0.0", 
    port: int = 8000, 
    reload: bool = False
):
    """Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload
    """
    uvicorn.run(
        "llamasum.api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    # Get port from command line or environment
    port = int(os.environ.get("PORT", 8000))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    
    # Run server
    run_server(port=port) 