# Writing Room architecture

## Boundary

The production artifact is a static browser application:

```text
Svelte components
    -> workspace state
        -> storage service -> IndexedDB / localStorage
        -> provider service -> OpenAI-compatible HTTP + SSE
```

There is no SvelteKit server route and no application backend. SvelteKit uses
the static adapter and hash routing, so the same output can be served by
`strata-server`, a reverse proxy, a desktop wrapper, or an ordinary static host.

## Layers

- `src/routes` composes the application and owns request lifecycle and
  cancellation.
- `src/lib/components` renders the library, manuscript, conversation, and
  provider dialog. Components do not make HTTP or storage calls.
- `src/lib/state` owns books, scenes, messages, provider selection, and
  debounced persistence.
- `src/lib/services/openai.ts` is the only inference-provider adapter.
- `src/lib/services/sse.ts` handles byte framing independently of OpenAI JSON.
- `src/lib/services/storage.ts` owns browser persistence.

The provider adapter accepts plain typed data and an injectable `fetch`
implementation. Tests therefore exercise the wire contract without launching
Strata or any other inference engine.

## Data ownership

One IndexedDB snapshot contains books, scenes, and per-book conversations. It
has an explicit schema version. Provider configuration is separate in
localStorage so resetting or changing a provider never mutates manuscripts.

The API key never enters a manuscript snapshot, URL, log statement, or export.
Markdown exports contain book text only.

## Inference request construction

Each request is constructed from:

1. the configurable collaborator instruction;
2. an optional, explicitly toggled snapshot of the book compass and current
   scene;
3. the exact visible conversation, in order;
4. the new author message.

No message is silently summarized, dropped, or rewritten. If a provider's
context limit is exceeded, its error is shown in the conversation. Context
management can become an explicit author-controlled feature later.

Streaming text is batched to one browser render per animation frame. This keeps
the transcript responsive without changing, delaying, or coalescing the text
stored in the completed assistant message.

## Deliberately deferred

- server-side accounts, synchronization, and collaboration;
- embeddings and retrieval;
- tool execution;
- image or document attachments;
- model loading and routing APIs;
- branching and revision comparison;
- embedded static assets in the Strata executable.

Those features must extend the browser-domain model or provider interface. They
must not introduce checks for Strata-specific endpoints into UI components.
