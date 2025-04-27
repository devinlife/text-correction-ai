#!/usr/bin/env python3
"""
OCR Text Correction Tool

This script processes OCR-extracted text files by using LangChain with Claude AI
to reconstruct and correct formatting issues while preserving content.
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ProcessingOptions:
    """Configuration options for text processing."""
    chunk_size: int = 4000
    chunk_overlap: int = 200
    max_concurrency: int = 3
    model: str = "claude-3-7-sonnet-20250219"
    temperature: float = 0.0
    max_tokens: int = 4096


class ChunkResult(TypedDict):
    """Type definition for chunk processing results."""
    index: int
    text: str
    success: bool


async def read_file(file_path: Path) -> str:
    """Read the OCR text file asynchronously and return its contents as a string."""
    try:
        return await asyncio.to_thread(Path(file_path).read_text, encoding='utf-8')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


async def write_output_file(input_file_path: Path, corrected_text: str) -> Path:
    """Write the corrected text to a new file asynchronously."""
    input_path = Path(input_file_path)
    output_path = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
    
    try:
        await asyncio.to_thread(output_path.write_text, corrected_text, encoding='utf-8')
        print(f"Corrected text saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)


def create_langchain_pipeline(options: ProcessingOptions, api_key: str) -> Any:
    """Create a LangChain correction pipeline with the given options."""
    # Initialize LLM
    llm = ChatAnthropic(
        model=options.model,
        temperature=options.temperature,
        anthropic_api_key=api_key,
        max_tokens=options.max_tokens,
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at correcting OCR text. Your task is to fix formatting issues in the provided text
                 while maintaining the original content as much as possible. Focus on correcting paragraph breaks,
                 removing unnecessary line breaks, and maintaining proper spacing. Do not add or remove content,
                 just correct the formatting. Return only the corrected text without any explanations or comments."""),
        ("human", "Please correct the formatting of this OCR text while preserving the content:\n\n{text}")
    ])
    
    # Create the processing chain
    chain = (
        {"text": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain


async def process_chunks(
    pipeline: Any, 
    chunks: List[str], 
    options: ProcessingOptions
) -> List[str]:
    """Process multiple chunks using the LangChain pipeline with controlled concurrency."""
    results: List[ChunkResult] = []
    semaphore = asyncio.Semaphore(options.max_concurrency)
    tasks = []
    
    async def process_single_chunk(chunk: str, index: int) -> ChunkResult:
        """Process a single chunk with semaphore to limit concurrency."""
        async with semaphore:
            print(f"Processing chunk {index+1}/{len(chunks)} ({len(chunk):,} characters)...")
            try:
                result = await pipeline.ainvoke(chunk)
                return {"index": index, "text": result, "success": True}
            except Exception as e:
                print(f"Error processing chunk {index+1}: {e}", file=sys.stderr)
                return {"index": index, "text": chunk, "success": False}
    
    # Create tasks for each chunk
    for i, chunk in enumerate(chunks):
        task = asyncio.create_task(process_single_chunk(chunk, i))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Sort results by index to maintain correct order
    sorted_results = sorted(results, key=lambda x: x["index"])
    return [result["text"] for result in sorted_results]


async def process_file(file_path: Path, api_key: str, options: ProcessingOptions) -> Path:
    """Process an entire file using LangChain for text splitting and correction."""
    # Read file
    ocr_text = await read_file(file_path)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split at paragraph boundaries first
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(ocr_text)
    print(f"Text split into {len(chunks)} chunks using LangChain's RecursiveCharacterTextSplitter")
    
    # Create processing pipeline
    pipeline = create_langchain_pipeline(options, api_key)
    
    # Process all chunks
    corrected_chunks = await process_chunks(pipeline, chunks, options)
    
    # Join the corrected chunks
    corrected_text = "\n\n".join(chunk.strip() for chunk in corrected_chunks if chunk)
    
    # Write output file
    return await write_output_file(file_path, corrected_text)


async def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='OCR text correction using LangChain with Claude AI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('file_path', type=Path, help='Path to the OCR text file')
    parser.add_argument('--api_key', help='Anthropic API key (optional if set in environment variable)')
    parser.add_argument('--chunk_size', type=int, default=4000, help='Maximum size of text chunks for processing')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Overlap between adjacent chunks')
    parser.add_argument('--max_concurrency', type=int, default=3, help='Maximum number of concurrent API calls')
    parser.add_argument('--model', default="claude-3-7-sonnet-20250219", help='Claude model to use')
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Anthropic API key not provided. Please set ANTHROPIC_API_KEY environment variable or use --api_key", 
              file=sys.stderr)
        sys.exit(1)
    
    # Create processing options
    options = ProcessingOptions(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_concurrency=args.max_concurrency,
        model=args.model
    )

    # Process the file
    await process_file(args.file_path, api_key, options)


if __name__ == "__main__":
    asyncio.run(main())