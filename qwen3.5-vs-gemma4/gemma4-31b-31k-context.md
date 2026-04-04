cat  /Users/albou/tmp/abstractframework/abstractcore/CHANGELOG.md | ollama run gemma4:31b --verbose "Summarize this in 5 sentences"
Thinking...
Changelog for "AbstractCore" (formerly "AbstractLLM").
Summarize the content in exactly 5 sentences.

    *   *V2.12.0 (Unreleased):* Huge focus on prompt caching (`CachedSession`, KV reuse), thinking/reasoning 
observability (thinking tags, reasoning property), and expanded model support (GPT-5 family, Claude 4.x, Gemma 4).
    *   *V2.11.x:* Portkey provider, skim tools for token-efficient web search, security policy, and MLX 
throughput benchmarking.
    *   *V2.10.x:* OpenRouter provider, OpenAI-compatible consolidation.
    *   *V2.9.0:* MCP (Model Context Protocol) integration, updated tools infrastructure, and new MoE models.
    *   *V2.8.0:* New architectures (Qwen3-VL, etc.), compression modes for history, and improved tool call 
parsing.
    *   *V2.6.x:* vLLM provider, OpenAI-compatible generic provider, and dynamic base URL support.
    *   *V2.5.x:* Native async/await support for high performance, interaction tracing, and glyph visual-text 
compression.
    *   *V2.4.x:* Major rebranding from "AbstractLLM" to "AbstractCore," unified token terminology, and a robust 
media handling system (PDF, Office docs).
    *   *V2.3.x:* Vector embeddings, high-level processing apps (Summarizer, Extractor, Judge), and a dedicated 
documentation website.
    *   *V2.2.x:* LLM-as-a-Judge (BasicJudge) and ONNX optimizations for embeddings.

    *   *Sentence 1: Evolution & Branding.* The project evolved from AbstractLLM to AbstractCore, establishing a 
unified API for a vast array of LLM providers including OpenAI, Anthropic, Ollama, and others.
    *   *Sentence 2: Performance & Async.* Significant performance gains were achieved through the implementation 
of native async/await support and prompt caching strategies for local and cloud-based models.
    *   *Sentence 3: Advanced Features (Reasoning/Thinking).* Recent updates introduced advanced observability for 
"thinking" and reasoning models, along with native structured output and MCP (Model Context Protocol) integration.
    *   *Sentence 4: Tooling & Media.* The ecosystem expanded to include sophisticated media handling for PDFs and 
Office documents, as well as a suite of processing applications like the BasicJudge and BasicSummarizer.
    *   *Sentence 5: Ecosystem & Hardware.* Broad hardware and model compatibility were prioritized, adding 
support for Apple Silicon (MLX), NVIDIA CUDA (vLLM), and a wide variety of state-of-the-art MoE and vision 
architectures.

    *   Check sentence count: 5.
    *   Check coverage: Rebranding (2.4.0), Performance (2.5.x/2.12.0), Reasoning/Thinking (2.12.0), Tools/Media 
(2.4.x), Hardware (2.6.x/2.12.0).
...done thinking.

AbstractCore (formerly AbstractLLM) has evolved into a comprehensive unified API supporting a vast array of LLM 
providers, including OpenAI, Anthropic, Ollama, and vLLM. Recent updates have introduced significant performance 
enhancements through native async/await support and advanced prompt caching strategies for both local and 
cloud-based models. The project has expanded its capabilities to include native structured outputs, MCP (Model 
Context Protocol) integration, and deep observability for "thinking" and reasoning models. A robust media handling 
system now enables the processing of PDFs and Office documents, complemented by a suite of high-level applications 
like the BasicJudge and BasicSummarizer. Finally, the library ensures broad hardware and model compatibility, 
specifically optimizing for Apple Silicon via MLX and NVIDIA GPUs via vLLM across a wide variety of SOTA 
architectures.

total duration:       6m19.730412208s
load duration:        7.485184167s
prompt eval count:    32110 token(s)
prompt eval duration: 4m1.908445375s
prompt eval rate:     132.74 tokens/s
eval count:           853 token(s)
eval duration:        2m9.917449177s
eval rate:            6.57 tokens/s