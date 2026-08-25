# Strata Writing Room

Strata Writing Room is a small, local-first Svelte application for authors who
want a language model beside the manuscript rather than in place of it.

It is deliberately independent of the Strata inference runtime. Any service
that implements the OpenAI-compatible Models and Chat Completions endpoints can
be used. Strata is the convenient local default, not a private dependency.

## What the first version includes

- books containing multiple scenes;
- a distraction-light manuscript editor with autosave and word counts;
- a separate, per-book model conversation;
- optional sharing of the active scene and book compass with the model;
- streamed responses with stop/cancellation;
- configurable base URL, API key, model, temperature, token limit, and system
  instructions;
- model discovery through `GET /models`;
- browser-local persistence and Markdown export;
- responsive desktop and mobile layouts.

Manuscripts and conversations are stored in IndexedDB. Provider settings are
stored in localStorage. There is no application backend or telemetry.

## Development

Node.js is a build and development dependency only. It is not required to serve
the compiled application.

```bash
cd web
npm ci
npm run dev
```

Open the URL printed by Vite. During development, relative `/v1` requests are
proxied to `http://127.0.0.1:8080`. Override the target without changing saved
provider settings:

```bash
STRATA_UI_DEV_SERVER_ORIGIN=http://127.0.0.1:8033 npm run dev
```

For Strata, start `strata-server` normally and select **Same-origin /v1** in the
provider dialog. For another provider, enter its complete API base URL, such as
`https://provider.example/v1`.

## Provider contract

The UI only requires:

```text
GET  {baseUrl}/models
POST {baseUrl}/chat/completions
```

Chat requests use the standard `model`, `messages`, `temperature`,
`max_tokens`, `stream`, and `stream_options.include_usage` fields. Streaming
responses use `text/event-stream`, OpenAI-style `choices[].delta.content`, and
the `[DONE]` sentinel.

The model field can also be entered manually when a compatible provider does
not expose `/models`. Standard usage is shown when available. Strata's optional
`timings.predicted_per_second` extension is displayed opportunistically and is
never required.

## Production build

```bash
npm run build
```

The static application is written to `web/build/`. Serve those files from any
static host. The cleanest deployment puts the UI and provider behind one
origin, serving the files at `/` and reverse-proxying `/v1` to the selected
inference service. This avoids CORS and keeps a local API key out of URLs.

The UI does not need Node, SvelteKit, or a database server after compilation.
Embedding the same static output into `strata-server` can be added as an
optional packaging mode without changing the frontend provider contract.

## Verification

```bash
npm run check      # Svelte and TypeScript diagnostics
npm run test       # provider, SSE, and utility tests
npm run build      # static production bundle
npm run test:e2e   # mocked-provider browser workflow
```

The browser test covers configuration, model discovery, manuscript editing,
scene context, streaming, metrics, IndexedDB reload, and Markdown export.

## Privacy note

A key stored in a browser can be read by anyone with access to that browser
profile and by code running on the same origin. Prefer a restricted key. For a
shared or internet-facing installation, place authentication, TLS, and secret
management in a trusted reverse proxy rather than distributing a broad cloud
key to browsers.

See [docs/architecture.md](docs/architecture.md) for the boundaries that keep
the application provider-neutral.
