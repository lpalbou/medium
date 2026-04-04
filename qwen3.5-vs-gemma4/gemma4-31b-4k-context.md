cat /Users/albou/tmp/abstractframework/abstractcore/README.md | ollama run gemma4:31b --verbose "Summarize this in 5 sentences"
Thinking...
*   Input: A comprehensive README for a Python library called "AbstractCore".
    *   Constraint: Summarize in exactly (or approximately) 5 sentences.

    *   *What is it?* A unified LLM interface ("Write once, run everywhere").
    *   *What does it do?* Provides a single `create_llm(...)` API for cloud (OpenAI, Anthropic) and local 
(Ollama, LMStudio, vLLM) providers.
    *   *Key Features:* Sync/async, streaming, Pydantic structured output, tool calling (passthrough by default), 
and multi-media input (images/audio/video) with a fallback system.
    *   *Advanced/Unique Features:* Glyph visual-text compression (rendering text to images for VLMs), an 
OpenAI-compatible gateway server, and integration with the AbstractFramework ecosystem (AbstractRuntime).
    *   *Installation:* Lightweight core with optional "extras" for specific providers or features (e.g., 
`[openai]`, `[media]`).

    *   *Sentence 1: Core Definition.* AbstractCore is a Python library that provides a unified interface for 
interacting with various cloud and local LLM providers via a single API.
    *   *Sentence 2: Key Functionality.* It supports synchronous and asynchronous calls, streaming, structured 
Pydantic outputs, and universal tool calling.
    *   *Sentence 3: Media/Advanced Features.* The library offers a sophisticated media handling system with 
policy-driven fallbacks and a unique "glyph" visual-text compression method to reduce token usage.
    *   *Sentence 4: Ecosystem/Tooling.* It includes an optional OpenAI-compatible gateway server and integrates 
with the broader AbstractFramework ecosystem for durable tool execution.
    *   *Sentence 5: Installation/Philosophy.* Designed to be lightweight, it allows users to install only the 
specific provider adapters and optional subsystems they need.

    *   S1: AbstractCore is a Python library that provides a unified `create_llm` API, allowing developers to 
switch between cloud and local LLM providers with minimal code changes.
    *   S2: It supports a wide range of essential features, including sync/async execution, streaming, structured 
Pydantic outputs, and universal tool calling.
    *   S3: Advanced capabilities include policy-driven media handling for images, audio, and video, as well as 
"glyph" visual-text compression to optimize long-document processing.
    *   S4: Additionally, the library offers an OpenAI-compatible gateway server and integrates with the 
AbstractFramework ecosystem for state persistence and tool execution.
    *   S5: To maintain a small footprint, the package uses an optional "extras" installation system so users only 
download the providers and subsystems they actually use.
...done thinking.

AbstractCore is a Python library that provides a unified `create_llm` API, allowing developers to switch between 
various cloud and local LLM providers with minimal code changes. It supports a wide range of essential features, 
including sync/async execution, streaming, structured Pydantic outputs, and universal tool calling. Advanced 
capabilities include policy-driven media handling for images, audio, and video, as well as "glyph" visual-text 
compression to optimize token usage for long documents. Additionally, the library offers an OpenAI-compatible 
gateway server and integrates with the broader AbstractFramework ecosystem for durable tool execution and state 
persistence. To maintain a small footprint, the package utilizes an optional "extras" installation system so users 
only download the specific providers and subsystems they need.

total duration:       1m29.885204791s
load duration:        5.334511458s
prompt eval count:    4173 token(s)
prompt eval duration: 29.77277875s
prompt eval rate:     140.16 tokens/s
eval count:           748 token(s)
eval duration:        54.523356317s
eval rate:            13.72 tokens/s