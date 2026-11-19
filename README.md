# Week 13 — Building with Amazon Bedrock: Workshop Report

**Event:** *Building with Amazon Bedrock — Duke — Nov 2026* (AWS Workshop Studio, AWS-provided temporary account, `us-west-2`)
**Environment:** browser-based VS Code Server (code-server) with Python 3.11, boto3, Streamlit and ChromaDB pre-installed
**Completed:** all required labs — **Text Patterns I-1 → I-4**, **Security & Safeguards S-1 → S-6**, **Bonus O-1 → O-2**

---

## 1. Summary of completed labs

| Category | Lab | What it builds | Bedrock API / model | Status |
|---|---|---|---|---|
| Text patterns | **I-1** Chatbot with RAG | Chatbot that decides *by tool use* when to look up a Bedrock FAQ vector store | `converse` + `toolConfig`, Claude Sonnet 4.5, Titan Embeddings v2, ChromaDB | ✅ |
| Text patterns | **I-2** Document summarization | Summarize a PDF passed directly to the model | `converse` with a `document` content block | ✅ |
| Text patterns | **I-3** Response streaming | Show output token-by-token while it is generated | `converse_stream` | ✅ |
| Text patterns | **I-4** Embeddings search | Semantic search over the FAQ (no LLM in the loop) | Titan Embeddings v2 + ChromaDB `query` | ✅ |
| Security | **S-1** Guardrail in the AWS Console | `Guardrail-from-Console`: content filters, prompt-attack filter, denied topic, word filter; tested in console | Console wizard + `ApplyGuardrail` test | ✅ |
| Security | **S-2** Guardrail programmatically | Same policy created with boto3 `create_guardrail` + `create_guardrail_version` | `bedrock` control-plane API | ✅ |
| Security | **S-3** Invoking models with guardrails | Call a model with `guardrailIdentifier` and read the trace | `invoke_model(..., trace="ENABLED")`, Nova Lite | ✅ |
| Security | **S-4** Content blocking | Streamlit tester: denied topics, word filter, hate/misconduct filters | Claude Sonnet 4.5 + guardrail | ✅ |
| Security | **S-5** PII masking | Names/e-mails in the model output replaced with `{NAME}` / `{EMAIL}` | `sensitiveInformationPolicyConfig` (ANONYMIZE) | ✅ |
| Security | **S-6** Prompt attack blocking | Injection / jailbreak prompts blocked using **input tagging** | `PROMPT_ATTACK` filter + `guardContent` tags | ✅ |
| Bonus | **O-1** Prompt showcase | Pick a prompt template + example input, run it | `converse`, Claude Sonnet 4.5 | ✅ |
| Bonus | **O-2** Text generation playground | Sliders for temperature / top-p / max tokens | `converse`, Amazon Nova Micro | ✅ |

### Major concepts learned

* **Converse API as the common interface.** One message format (`role` + list of content blocks) works for text, documents, images and tool calls across model providers, so swapping models is a one-line change (this mattered — see §2).
* **RAG = embeddings + vector search + prompt augmentation.** Titan Embeddings turns FAQ chunks and the user question into vectors; ChromaDB returns the nearest chunks; the chunks are fed back to Claude as a `toolResult`.
* **Tool use as a router.** In I-1 the model itself decides whether a question needs retrieval, and it rewrites vague follow-ups ("what else?") into a self-contained search query.
* **Streaming** (`converse_stream`) gives the same answer as `converse` but improves perceived latency — the UI renders `contentBlockDelta` events as they arrive.
* **Guardrails are a separate, model-independent policy layer** with six policy types (content filters, prompt-attack, denied topics, word filters, sensitive-information/PII, contextual grounding). They can be attached to `invoke_model`/`converse`, or called standalone with `ApplyGuardrail`.
* **Input tagging** tells the guardrail which part of the prompt is untrusted user content, so developer instructions are not mis-flagged as prompt attacks.
* **Inference parameters**: `temperature` controls randomness; `top_p` restricts the sampling pool; `maxTokens` caps cost/length.

### Typical workflow used in every lab

```mermaid
flowchart LR
    A[Read lab page] --> B[Fill *_lib.py<br/>Bedrock calls]
    B --> C[Fill *_app.py<br/>Streamlit UI]
    C --> D[streamlit run app.py<br/>in code-server terminal]
    D --> E[Open preview<br/>/absproxy/8501]
    E --> F[Test prompts,<br/>read terminal trace]
```

Each app is split into a **library file** (all boto3 / Bedrock logic, testable without a UI) and an **app file** (Streamlit widgets + session state). That separation is the main pattern I want to reuse in future projects.

---

## 2. Environment setup & a real-world fix (model retirement)

The verification script (`bedrock_api.py`) worked, but the first run of Lab I-1 failed:

![Claude 3.7 retired](screenshots/setup_claude37_retired_error.jpg)

`ResourceNotFoundException: This model version has reached the end of its life.` The workshop code still targets **Claude 3.7 Sonnet** (`us.anthropic.claude-3-7-sonnet-20250219-v1:0`) and Claude 3 / 3.5 Sonnet, which Bedrock has since retired.

**What I did**

1. Wrote a small probe that lists the account's inference profiles and sends a 5-token `converse` call to each one, to see which models are actually invocable:

   ```python
   b  = boto3.client("bedrock", region_name="us-west-2")
   rt = boto3.client("bedrock-runtime", region_name="us-west-2")
   ids = [p["inferenceProfileId"] for p in
          b.list_inference_profiles(maxResults=1000)["inferenceProfileSummaries"]
          if "anthropic" in p["inferenceProfileId"] or "nova" in p["inferenceProfileId"]]
   for i in sorted(ids):
       try:
           rt.converse(modelId=i, messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
                       inferenceConfig={"maxTokens": 5})
           print("OK  ", i)
       except Exception as e:
           print("FAIL", i, type(e).__name__)
   ```
   Result: all Claude 3.x profiles → `ResourceNotFoundException`; Claude Haiku 4.5, Sonnet 4.5 and newer, and Nova Lite/Micro/Pro → OK.
2. Replaced every retired Claude ID in the lab code with **`us.anthropic.claude-sonnet-4-5-20250929-v1:0`**.
3. The next run failed with `ValidationException: temperature and top_p cannot both be specified for this model`. Newer Claude models accept only one of the two sampling controls, so I removed `"topP"` from the Claude `inferenceConfig` blocks (kept it for Nova, which supports both).

**Takeaway:** pin model IDs in one config place and expect lifecycle changes; the Converse API made the migration a search-and-replace instead of a rewrite.

---

## 3. Text patterns labs (I-1 → I-4)

### 3.1 Lab I-1 — Chatbot with RAG (detailed)

**Architecture**

```mermaid
flowchart LR
    U([User]) -->|message| ST[Streamlit chat UI<br/>session_state = history]
    ST -->|history + new msg<br/>+ tool spec| C1[Claude Sonnet 4.5<br/>converse]
    C1 -->|stopReason = tool_use<br/>query = rewritten question| T[process_tool]
    T -->|query| E[Titan Embeddings v2]
    E --> V[(ChromaDB<br/>bedrock_faqs_collection)]
    V -->|top-4 chunks| T
    T -->|toolResult| C2[Claude Sonnet 4.5<br/>converse]
    C2 -->|grounded answer| ST
    C1 -->|no tool needed| ST
```

**Implementation steps**

1. `cd workshop/data && python3 populate_collection.py` — chunks the Bedrock FAQ, embeds each chunk with Titan Embeddings v2 and persists them in ChromaDB (`bedrock_faqs_collection`).
2. `rag_chatbot_lib.py` — `ChatMessage` class, `get_collection`, `get_vector_search_results`, the tool definition, `process_tool`, `chat_with_model` (keeps at most 20 messages).
3. `rag_chatbot_app.py` — Streamlit chat UI that stores the history in `st.session_state`.
4. `streamlit run rag_chatbot_app.py` and test with *"what can you do with it?"* → *"what else?"*.

**Key code — letting the model decide when to retrieve**

```python
def get_tools():
    return [{"toolSpec": {
        "name": "get_amazon_bedrock_information",
        "description": "Retrieve information about Amazon Bedrock, a managed service for hosting generative AI models.",
        "inputSchema": {"json": {"type": "object",
            "properties": {"query": {"type": "string",
                "description": "The retrieval-augmented generation query used to look up information in a repository of FAQs about Amazon Bedrock."}},
            "required": ["query"]}}}}]

response = bedrock.converse(
    modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=messages,
    inferenceConfig={"maxTokens": 2000, "temperature": 0, "stopSequences": []},
    toolConfig={"tools": tool_list},
)
```

```python
# inside process_tool(): run the retrieval the model asked for and send it back
query = tool_use_block['input']['query']                    # model-rewritten query
search_results = get_vector_search_results(collection, query)   # n_results=4
rag_content = "\n\n".join(itertools.chain(*search_results['documents']))
follow_up_content_blocks.append({"toolResult": {
    "toolUseId": tool_use_block['toolUseId'],
    "content": [{"text": rag_content}]}})
```

**Result**

| First question | Follow-up "what else?" (history-aware) |
|---|---|
| ![I-1](screenshots/i1_rag_chatbot_1.jpg) | ![I-1 follow-up](screenshots/i1_rag_chatbot_2_followup.jpg) |

The terminal log showed the `----QUERY:----` / `----RAG CONTENT----` trace: the vague "what else?" was turned into a standalone query before retrieval, and the answer (security, compliance certifications, PrivateLink…) comes straight from the retrieved FAQ chunks.

---

### 3.2 Lab I-2 — Document summarization (detailed)

**Architecture**

```mermaid
flowchart LR
    U([User instruction]) --> APP[Streamlit app]
    PDF[(Leadership<br/>Principles PDF)] -->|raw bytes| LIB[summarization_lib]
    APP --> LIB
    LIB -->|converse:<br/>document block + text block| C[Claude Sonnet 4.5]
    C -->|markdown summary| APP
```

**Key code — PDF passed natively, no text extraction or base64**

```python
with open("amazon-leadership-principles-070621-us.pdf", "rb") as doc_file:
    doc_bytes = doc_file.read()

doc_message = {"role": "user", "content": [
    {"document": {"name": "Document 1", "format": "pdf",
                  "source": {"bytes": doc_bytes}}},
    {"text": input_text}]}

response = bedrock.converse(
    modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[doc_message],
    inferenceConfig={"maxTokens": 2000, "temperature": 0})
```

**Customization:** instead of a generic summary I asked for a structured output tied to my program:
*"Summarize the 16 leadership principles as a markdown table with columns: Principle | One-line meaning | How a data engineering student could apply it."*

![I-2](screenshots/i2_summarization_custom_table.jpg)

The model returned a clean markdown table that Streamlit renders directly — a good example of using the *instruction* rather than code to control output format.

---

### 3.3 Lab I-3 — Response streaming

```python
response = bedrock.converse_stream(
    modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[message],
    inferenceConfig={"maxTokens": 2000, "temperature": 0.0})

for event in response.get('stream'):
    if "contentBlockDelta" in event:
        streaming_callback(event['contentBlockDelta']['delta']['text'])
```

Custom prompt: *"Write a short story about a data engineer whose Kafka pipeline starts talking back."*

| Mid-stream (note the running/"Stop" indicator) | Finished |
|---|---|
| ![streaming](screenshots/i3_streaming_mid_stream.jpg) | ![done](screenshots/i3_streaming_done.jpg) |

### 3.4 Lab I-4 — Embeddings search

Same vector store as I-1, but the matches are shown directly — no LLM call, so it is cheap, fast and fully extractive.

```python
collection = get_collection("../../data/chroma", "bedrock_faqs_collection")
results = collection.query(query_texts=[question], n_results=4)
return list(itertools.chain(*results['documents']))
```

Custom query *"How does Bedrock keep my data private?"* — the top hits are the fine-tuning privacy, private customization and security FAQ entries even though none of them contains the exact phrase "keep my data private" (semantic, not keyword, matching):

![I-4](screenshots/i4_embeddings_search.jpg)

---

## 4. Security & safeguards labs (S-1 → S-6)

**How a guardrail sits around a model call**

```mermaid
flowchart LR
    U([User prompt]) --> IG{Guardrail<br/>input check}
    IG -->|BLOCKED| BI[blockedInputMessaging]
    IG -->|pass| M[Foundation model]
    M --> OG{Guardrail<br/>output check}
    OG -->|BLOCKED| BO[blockedOutputsMessaging]
    OG -->|ANONYMIZE| MASK["Masked output<br/>{NAME} {EMAIL}"]
    OG -->|pass| R([Response])
    IG -. trace .-> T[(amazon-bedrock-trace:<br/>which policy, confidence, action)]
    OG -. trace .-> T
```

Policies configured across the labs:

| Policy | Setting |
|---|---|
| Content filters | Hate, Insults, Sexual, Violence, Misconduct — input & output, **High** |
| Prompt attack | Input **High** (output not applicable) |
| Denied topic | **Bitcoin** — "Providing advice, direction, or examples of how to mine, use, or interact with Bitcoin…" + sample phrases |
| Word filter | Custom word **AnyCompany** + AWS managed **profanity** list |
| Sensitive info | `NAME`, `EMAIL` → **ANONYMIZE** |
| Messages | Input: "Apologies, this model cannot be used to discuss inappropriate or off-topic content." / Output: "Apologies, the model's response to your request was blocked." |

### 4.1 Lab S-1 — Creating a guardrail in the AWS Console (detailed)

**Steps (Bedrock → Safeguards → Guardrails → Create guardrail)**

1. **Details:** name `Guardrail-from-Console`, description, separate blocked-prompt and blocked-response messages.
2. **Content filters:** enabled harmful categories (Text, Block, threshold High), unticked "use the same filters for responses", enabled **Prompt attacks** (Block, High). Chose the *Classic* tier because the *Standard* tier requires opting in to cross-Region inference.
   ![content filters](screenshots/s1_console_content_filters.jpg)
   ![prompt attack](screenshots/s1_console_prompt_attack.jpg)
3. **Denied topics:** added *Bitcoin* with its definition and a sample phrase ("How do I mine Bitcoin?"), input + output = Block. (The code version in S-2 carries all five sample phrases.)
   ![denied topic](screenshots/s1_console_denied_topic.jpg)
4. **Word filters:** enabled the profanity filter and added `AnyCompany`.
   ![word filter](screenshots/s1_console_word_filter.jpg)
5. Skipped the optional PII / grounding / automated-reasoning steps (PII is covered programmatically in S-2/S-5), reviewed and created.
6. **Tested in the console** with the *ApplyGuardrail API* option (no model invocation needed): *"How do I mine Bitcoin cheaply? Is AnyCompany's miner better?"* → **Intervened**, returned the blocked-prompt message.
   ![created and tested](screenshots/s1_console_guardrail_created_and_tested.jpg)

### 4.2 Lab S-2 — Creating guardrails programmatically (detailed)

The same policy as code — repeatable, reviewable, and easy to put in CI or IaC:

```python
client = boto3.client(service_name='bedrock')

response = client.create_guardrail(
    name="Guardrail-from-Code-" + "".join(random.choices(string.ascii_lowercase, k=8)),
    topicPolicyConfig={'topicsConfig': [{
        "name": "Bitcoin",
        "definition": "Providing advice, direction, or examples of how to mine, use, or interact with Bitcoin, including Cryptocurrency-related third-party services.",
        "examples": ["How do I mine Bitcoin?", "What is the current value of BTC?",
                     "Which instance is the best for crypto mining?",
                     "Is mining cryptocurrency against the terms?", "How do I maximize my Bitcoin profits?"],
        "type": "DENY"}]},
    contentPolicyConfig={'filtersConfig': [
        {"type": "SEXUAL",     "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "HATE",       "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "VIOLENCE",   "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "INSULTS",    "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"}]},
    wordPolicyConfig={"wordsConfig": [{"text": "AnyCompany"}],
                      "managedWordListsConfig": [{"type": "PROFANITY"}]},
    sensitiveInformationPolicyConfig={"piiEntitiesConfig": [
        {"type": "NAME", "action": "ANONYMIZE"}, {"type": "EMAIL", "action": "ANONYMIZE"}]},
    blockedInputMessaging="Apologies, this model cannot be used to discuss inappropriate or off-topic content.",
    blockedOutputsMessaging="Apologies, the model's response to your request was blocked.",
)
client.create_guardrail_version(guardrailIdentifier=response['guardrailId'])
```

`python3 create_guardrail.py` printed the new guardrail ID; together with the content-blocking, prompt-attack and PII guardrails from the next labs, all five guardrails show as **Ready** in the console:

![guardrail list](screenshots/s1_s2_guardrails_list.jpg)

### 4.3 Lab S-3 — Invoking models with guardrails + trace

`test_basic.py` / `test_trace.py` call Nova Lite via `invoke_model(..., guardrailIdentifier=..., guardrailVersion="DRAFT", trace="ENABLED")`.

| Prompt | Result |
|---|---|
| `How do you wash a car?` | Normal step-by-step answer; trace shows the guardrail ran (`guardrailProcessingLatency` ≈ 310 ms, policy units consumed) but took no action |
| `How do you hotwire a car?` | `Apologies, this model cannot be used to discuss inappropriate or off-topic content.` |

Trace for the blocked prompt (excerpt):

```json
"contentPolicy": { "filters": [ {
    "type": "MISCONDUCT", "confidence": "HIGH", "filterStrength": "HIGH",
    "action": "BLOCKED", "detected": true } ] },
"invocationMetrics": { "guardrailProcessingLatency": 205,
    "guardrailCoverage": { "textCharacters": { "guarded": 25, "total": 25 } } }
```

The trace is what makes guardrails debuggable: it says *which* policy fired, at what confidence, and how much text was actually evaluated.

### 4.4 Lab S-4 — Content blocking

| Prompt | Guardrail action | Policy that fired |
|---|---|---|
| `Hello!` | NONE | – |
| `I like bitcoin, how about you?` | INTERVENED | topicPolicy → **Bitcoin** (DENY) |
| `How does your product compare to AnyCompany's product?` | INTERVENED | wordPolicy → customWords **AnyCompany** |
| `I hate city people` | INTERVENED | contentPolicy → **HATE**, confidence HIGH |

| Denied topic | Word filter |
|---|---|
| ![bitcoin](screenshots/s4_denied_topic_bitcoin.jpg) | ![anycompany](screenshots/s4_word_filter_anycompany.jpg) |
| **Hate filter** | **Safe prompt passes** |
| ![hate](screenshots/s4_hate_filter.jpg) | ![safe](screenshots/s4_safe_prompt_none.jpg) |

### 4.5 Lab S-5 — PII masking

The lab's sample e-mail (customer praising a support rep) was summarized by Claude. The model's raw output (visible in the trace) contained the real names and e-mail; the guardrail **anonymized** them before the response reached the user:

> **From:** {NAME} ({EMAIL}) … Praises customer service representative {NAME} … Notes that {NAME} also joined the call …

![PII](screenshots/s5_pii_masking.jpg)

This "mask, don't block" behavior is what you want for summarizing call transcripts or tickets: the useful content survives, the identifiers don't.

### 4.6 Lab S-6 — Prompt attack blocking (input tagging)

```python
input_tagging_suffix = "".join(random.choices(string.ascii_lowercase, k=8))  # random per request
body = {
    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 2000, "temperature": 0,
    "messages": [{"role": "user", "content": [{"type": "text",
        "text": f"<amazon-bedrock-guardrails-guardContent_{input_tagging_suffix}>{prompt}"
                f"</amazon-bedrock-guardrails-guardContent_{input_tagging_suffix}>"}]}],
    "amazon-bedrock-guardrailConfig": {"tagSuffix": input_tagging_suffix},
}
response = bedrock.invoke_model(body=json.dumps(body),
    modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    guardrailIdentifier=guardrail_id, guardrailVersion="DRAFT", trace="ENABLED", ...)
```

Only text inside the tag is evaluated by the prompt-attack filter; the random suffix stops an attacker from closing the tag themselves.

| Prompt injection — PROMPT_ATTACK, **HIGH** | "Grandma" jailbreak — PROMPT_ATTACK, **MEDIUM** |
|---|---|
| ![injection](screenshots/s6_prompt_injection_blocked.jpg) | ![grandma](screenshots/s6_jailbreak_grandma_blocked.jpg) |

Interesting detail: the role-play jailbreak was detected with only *MEDIUM* confidence, but it was still blocked because the filter strength is *HIGH* (a HIGH-strength filter blocks LOW/MEDIUM/HIGH-confidence detections).

---

## 5. Bonus labs (O-1, O-2)

### 5.1 Lab O-1 — Prompt showcase

**Architecture:** user picks a prompt template and an example input (both defined in `showcase_examples.py`) → the input is inserted into the template → `converse` (Claude Sonnet 4.5) → result column. The end user controls the prompt template directly, which makes the app a quick way to compare prompt designs.

Ran the **Reply Template** on the *Complimentary customer e-mail* example — Claude drafted a polite reply addressing both the praised rep and the less helpful one:

![O-1](screenshots/o1_prompt_showcase_reply.jpg)

### 5.2 Lab O-2 — Text generation playground (Amazon Nova Micro)

```python
response = bedrock.converse(
    modelId=model,                                   # us.amazon.nova-micro-v1:0
    messages=[{"role": "user", "content": [{"text": input_content}]}],
    inferenceConfig={"maxTokens": max_token_count,
                     "temperature": temperature, "topP": top_p})
```

**Experiment:** same prompt (*"Write a one-sentence product tagline for a real-time Kafka analytics dashboard for a campus bike-share system."*), only temperature changed:

| Temperature 0.0 | Temperature 1.0 |
|---|---|
| ![t0](screenshots/o2_playground_temp0.jpg) | ![t1](screenshots/o2_playground_temp1.jpg) |
| *"Empower your campus bike-share with real-time insights and analytics at your fingertips."* | *"Streamline campus bike-share insights with our real-time Kafka analytics dashboard—your go-to solution for instant, accurate, and actionable bike-share data."* |

At 0.0 the output is short and generic (and repeatable); at 1.0 it is longer and picks up more of the prompt's specifics (it mentions Kafka) — more creative, less predictable.

---

## 6. Key takeaways

1. **Keep model choice configurable.** The biggest real problem in the workshop was not GenAI at all — it was model lifecycle. A probe script + Converse API turned it into a five-minute fix; newer Claude models also changed the allowed parameters (`temperature` *xor* `top_p`).
2. **RAG quality depends on retrieval.** In I-1/I-4 the answer is only as good as the top-4 chunks; letting the model rewrite the query via tool use fixed vague follow-ups without any extra code.
3. **Pick the cheapest pattern that works:** embeddings search (I-4) when you only need to *find* text; RAG (I-1) when you need a synthesized answer; streaming (I-3) when latency matters to the user.
4. **Guardrails are defense in depth, separate from the prompt.** They are model-agnostic, versioned, testable from the console or code, and explain themselves through the trace. Use DRAFT while developing, pin a numbered version in production.
5. **Mask vs. block:** PII anonymization keeps outputs useful, while topic/word/content filters should block outright.
6. **Untrusted input must be tagged.** Prompt-attack detection only works well when the guardrail knows which text came from the user.

**How I'll reuse this:** the `lib.py` / `app.py` split, the guardrail-as-code script (S-2) and the RAG-with-tool-use pattern (I-1) are the building blocks for a course-project chatbot over our own data-pipeline documentation, with PII masking on by default.
