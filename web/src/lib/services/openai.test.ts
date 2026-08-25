import { describe, expect, it, vi } from 'vitest';
import type { ProviderSettings } from '$lib/types';
import { listModels, streamCompletion } from './openai';

const settings: ProviderSettings = {
  baseUrl: 'http://writer.test/v1/',
  apiKey: 'secret',
  model: 'novelist',
  temperature: 0.75,
  maxTokens: 600,
  systemPrompt: 'Be a careful editor.'
};

function streamResponse(records: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const record of records) controller.enqueue(encoder.encode(record));
        controller.close();
      }
    }),
    { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
  );
}

describe('OpenAI-compatible provider service', () => {
  it('discovers and sorts models without provider-specific metadata', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          object: 'list',
          data: [
            { id: 'zeta', owned_by: 'local' },
            { id: 'alpha', owned_by: 'remote' },
            { id: 42 }
          ]
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    );

    await expect(listModels(settings, fetcher)).resolves.toEqual([
      { id: 'alpha', ownedBy: 'remote' },
      { id: 'zeta', ownedBy: 'local' }
    ]);
    expect(fetcher).toHaveBeenCalledWith(
      'http://writer.test/v1/models',
      expect.objectContaining({ headers: expect.any(Headers) })
    );
    const headers = fetcher.mock.calls[0][1]?.headers as Headers;
    expect(headers.get('Authorization')).toBe('Bearer secret');
  });

  it('streams text and maps both standard usage and optional Strata timings', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      streamResponse([
        'data: {"choices":[{"delta":{"role":"assistant","content":"Once "}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"upon a time."}}]}\n\n',
        'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24},"timings":{"predicted_per_second":12.5}}\n\n',
        'data: [DONE]\n\n'
      ])
    );
    const onText = vi.fn();
    const onUsage = vi.fn();
    const onTimings = vi.fn();

    const result = await streamCompletion({
      settings,
      messages: [
        { id: 'm1', role: 'user', content: 'Continue.', createdAt: '2026-01-01', status: 'complete' }
      ],
      sceneContext: 'The door opened.',
      signal: new AbortController().signal,
      callbacks: { onText, onUsage, onTimings },
      fetcher
    });

    expect(result.content).toBe('Once upon a time.');
    expect(result.usage?.totalTokens).toBe(24);
    expect(result.timings?.predictedPerSecond).toBe(12.5);
    expect(onText.mock.calls.flat()).toEqual(['Once ', 'upon a time.']);
    expect(onUsage).toHaveBeenCalledWith({ promptTokens: 20, completionTokens: 4, totalTokens: 24 });

    const [url, init] = fetcher.mock.calls[0];
    expect(url).toBe('http://writer.test/v1/chat/completions');
    const body = JSON.parse(String(init?.body));
    expect(body).toMatchObject({ model: 'novelist', temperature: 0.75, max_tokens: 600, stream: true });
    expect(body.messages).toEqual([
      { role: 'system', content: 'Be a careful editor.' },
      expect.objectContaining({ role: 'system', content: expect.stringContaining('The door opened.') }),
      { role: 'user', content: 'Continue.' }
    ]);
  });

  it('surfaces OpenAI-shaped provider errors', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: 'Unknown model' } }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      })
    );
    await expect(listModels(settings, fetcher)).rejects.toThrow('Unknown model');
  });
});
