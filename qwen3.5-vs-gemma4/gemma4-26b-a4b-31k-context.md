cat  /Users/albou/tmp/abstractframework/abstractcore/CHANGELOG.md | ollama run gemma4:26b --verbose "Summarize this in 5 sentences"
Thinking...
A long, detailed changelog for a project called "AbstractCore" (formerly "AbstractLLM").
Summarize the changelog in exactly 5 sentences.

    *   *Core Identity:* AbstractCore is a multi-provider LLM framework/orchestration library (supporting OpenAI, 
Anthropic, Ollama, LMStudio, MLX, HuggingFace, vLLM, etc.).
    *   *Key Feature Evolution (Versions 2.4.0 to 2.12.0):*
        *   *Architecture:* Rebranded from AbstractLLM to AbstractCore; introduced a centralized provider registry 
and unified interface.
        *   *Capabilities:* Major focus on "thinking" (reasoning) models, prompt caching (KV mode, box caching), 
and multimodal support (vision, audio, video, and text-based glyph compression).
        *   *Tools & MCP:* Integrated Model Context Protocol (MCP), enhanced tool calling (structured output, tool 
syntax rewriting), and improved web/file tools (skimming, extraction).
        *   *Performance/Efficiency:* Massive focus on async support (7x speedups), prompt caching for large 
contexts, and specialized optimizations for Apple Silicon (MLX) and NVIDIA (vLLM).
        *   *Infrastructure/DevEx:* Robust configuration management, centralized logging, advanced evaluation 
(LLM-as-a-judge), and a professional documentation site/web presence.
        *   *Deployment:* Expanded server capabilities (OpenAI-compatible endpoints, streaming, media handling) 
and better installation extras for different platforms (Linux/Windows/macens).

    *   *Sentence 1: Core Identity and Evolution.* AbstractCore evolved from AbstractLLM into a robust, unified 
framework for managing diverse LLM providers like OpenAI, Anthropic, and local engines like Ollama and MLX.
    *   *Sentence 2: Capability Expansion (Reasoning/Multimodal).* The project significantly expanded its 
capabilities to include advanced support for reasoning models, multi-modal processing (vision, audio, video), and 
sophisticated prompt caching strategies.
    *   *Sentence 3: Tooling and Protocol.* Major architectural updates introduced Model Context Protocol (MCP) 
integration, enhanced tool-calling reliability with structured outputs, and advanced web-skimming utilities.
    *   *Sentence 4: Performance and Scalations.* Significant performance improvements were achieved through 
native async support, optimized hardware-specific backends for Apple Silicon and NVIDIA GPUs, and specialized 
embedding management.
    *   *Sentence 5: Ecosystem and Infrastructure.* Finally, the ecosystem matured with professional 
documentation, a centralized configuration system, and advanced "LLM-as-a-judge" evaluation capabilities for 
automated assessment.

    *   *Sentence 1:* AbstractCore has evolved from AbstractLLM into a comprehensive, unified framework for 
orchestrating diverse LLM providers, ranging from cloud APIs like OpenAI and Anthropic to local engines like 
Ollama and MLX.
    *   *Sentence 2:* Significant architectural advancements include the introduction of Model Context Protocol 
(MCP) integration, advanced prompt caching for large contexts, and robust support for reasoning and multimodal 
models.
    
    *   *Sentence 3:* The project dramatically improved performance through native async support, 
hardware-optimized backends for Apple Silicon and NVIDIA GPUs, and efficient tool-calling mechanisms with 
structured outputs.
    
    *   *Sentence 4:* Tooling and data handling were significantly enhanced with the addition of sophisticated 
web-skimming, advanced file extraction, and specialized visual-text compression techniques.
    
    *   *Sentence 5:* Finally, the ecosystem matured with the implementation of a centralized configuration 
system, professional documentation, and "LLM-as-a-judge" capabilities for automated, high-quality assessments.

    1. Identity/Scope.
    2. Capabilities (Reasoning/MCP/Caching).
    3. Performance (Async/Hardware).
    4. Tooling/Data (Web/File/Vision).
    5. Infrastructure (Config/Judge/Docs).

    *Self-Correction during final review:* The user asked for 5 sentences. I will ensure the tone is professional 
and covers the breadth of the provided text.

    *Draft:*
    AbstractCore has transitioned from AbstractLLM into a unified orchestration framework supporting a vast array 
of providers, including OpenAI, Anthropic, Ollama, and MLX. The project has significantly expanded its 
capabilities to include advanced reasoning model control, sophisticated prompt caching, and multimodal processing 
for vision, audio, and video. Major architectural updates introduced Model Context Protocol (MCP) integration and 
highly reliable, structured tool-calling mechanisms. Performance was drastically boosted through native 
asynchronous support and hardware-specific optimizations for both Apple Silicon and NVIDIA GPU environments. 
Finally, the ecosystem matured with centralized configuration management, advanced "LLM-as-a-judge" evaluation 
tools, and a professionalized documentation infrastructure.
...done thinking.

AbstractCore has transitioned from AbstractLLM into a unified orchestration framework supporting a vast array of 
providers, including OpenAI, Anthropic, Ollama, and MLX. The project has significantly expanded its capabilities 
to include advanced reasoning model control, sophisticated prompt caching, and multimodal processing for vision, 
audio, and video. Major architectural updates introduced Model Context Protocol (MCP) integration and highly 
reliable, structured tool-calling mechanisms. Performance was drastically boosted through native asynchronous 
support and hardware-specific optimizations for both Apple Silicon and NVIDIA GPU environments. Finally, the 
ecosystem matured with centralized configuration management, advanced "LLM-as-a-judge" evaluation tools, and a 
professionalized documentation infrastructure.

total duration:       1m2.155024625s
load duration:        3.886128084s
prompt eval count:    32110 token(s)
prompt eval duration: 32.834638625s
prompt eval rate:     977.93 tokens/s
eval count:           1161 token(s)
eval duration:        24.950477922s
eval rate:            46.53 tokens/s