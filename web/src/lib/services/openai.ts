import type {
  ChatMessage,
  CompletionResult,
  CompletionTimings,
  CompletionUsage,
  ModelOption,
  ProviderSettings,
  StreamCallbacks
} from '$lib/types';
import { readSse } from './sse';

interface OpenAiErrorBody {
  error?: { message?: string };
  message?: string;
}

interface OpenAiChunk {
  choices?: Array<{
    delta?: { content?: string | null };
    text?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  timings?: {
    prompt_per_second?: number;
    predicted_per_second?: number;
  };
}

function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) throw new Error('API base URL is required.');
  return trimmed.replace(/\/+$/u, '');
}

function headersFor(settings: ProviderSettings): Headers {
  const headers = new Headers({
    Accept: 'application/json',
    'Content-Type': 'application/json'
  });
  if (settings.apiKey.trim()) headers.set('Authorization', `Bearer ${settings.apiKey.trim()}`);
  return headers;
}

async function responseError(response: Response): Promise<Error> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as OpenAiErrorBody;
    message = body.error?.message || body.message || message;
  } catch {
    const text = await response.text().catch(() => '');
    if (text.trim()) message = text.trim();
  }
  return new Error(message || 'The provider rejected the request.');
}

function mapUsage(chunk: OpenAiChunk): CompletionUsage | undefined {
  if (!chunk.usage) return undefined;
  return {
    promptTokens: chunk.usage.prompt_tokens,
    completionTokens: chunk.usage.completion_tokens,
    totalTokens: chunk.usage.total_tokens
  };
}

function mapTimings(chunk: OpenAiChunk): CompletionTimings | undefined {
  if (!chunk.timings) return undefined;
  return {
    promptPerSecond: chunk.timings.prompt_per_second,
    predictedPerSecond: chunk.timings.predicted_per_second
  };
}

export async function listModels(
  settings: ProviderSettings,
  fetcher: typeof fetch = fetch
): Promise<ModelOption[]> {
  const response = await fetcher(`${normalizeBaseUrl(settings.baseUrl)}/models`, {
    headers: headersFor(settings)
  });
  if (!response.ok) throw await responseError(response);

  const body = (await response.json()) as {
    data?: Array<{ id?: unknown; owned_by?: unknown }>;
  };
  if (!Array.isArray(body.data)) throw new Error('Provider returned an invalid model list.');

  return body.data
    .filter((model): model is { id: string; owned_by?: string } => typeof model.id === 'string')
    .map((model) => ({ id: model.id, ownedBy: model.owned_by }))
    .sort((left, right) => left.id.localeCompare(right.id));
}

export interface StreamCompletionInput {
  settings: ProviderSettings;
  messages: ChatMessage[];
  sceneContext?: string;
  signal: AbortSignal;
  callbacks: StreamCallbacks;
  fetcher?: typeof fetch;
}

export async function streamCompletion(input: StreamCompletionInput): Promise<CompletionResult> {
  const { settings, messages, sceneContext, signal, callbacks } = input;
  const fetcher = input.fetcher ?? fetch;
  const requestMessages: Array<{ role: string; content: string }> = [];

  if (settings.systemPrompt.trim()) {
    requestMessages.push({ role: 'system', content: settings.systemPrompt.trim() });
  }
  if (sceneContext?.trim()) {
    requestMessages.push({
      role: 'system',
      content:
        'The author has shared the current scene below as working context. Treat it as their text; do not claim authorship or rewrite it unless asked.\n\n<current_scene>\n' +
        sceneContext.trim() +
        '\n</current_scene>'
    });
  }
  requestMessages.push(
    ...messages.map((message) => ({ role: message.role, content: message.content }))
  );

  const response = await fetcher(`${normalizeBaseUrl(settings.baseUrl)}/chat/completions`, {
    method: 'POST',
    headers: headersFor(settings),
    signal,
    body: JSON.stringify({
      model: settings.model,
      messages: requestMessages,
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
      stream: true,
      stream_options: { include_usage: true }
    })
  });

  if (!response.ok) throw await responseError(response);
  if (!response.body) throw new Error('Provider returned an empty streaming response.');

  let content = '';
  let usage: CompletionUsage | undefined;
  let timings: CompletionTimings | undefined;

  for await (const event of readSse(response.body, signal)) {
    if (event.data === '[DONE]') break;

    let chunk: OpenAiChunk;
    try {
      chunk = JSON.parse(event.data) as OpenAiChunk;
    } catch {
      throw new Error('Provider returned malformed streaming JSON.');
    }

    const text = chunk.choices?.[0]?.delta?.content ?? chunk.choices?.[0]?.text ?? '';
    if (text) {
      content += text;
      callbacks.onText(text);
    }
    const nextUsage = mapUsage(chunk);
    if (nextUsage) {
      usage = nextUsage;
      callbacks.onUsage?.(nextUsage);
    }
    const nextTimings = mapTimings(chunk);
    if (nextTimings) {
      timings = nextTimings;
      callbacks.onTimings?.(nextTimings);
    }
  }

  return { content, usage, timings };
}
