import { describe, expect, it } from 'vitest';
import { readSse } from './sse';

function byteStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    }
  });
}

describe('readSse', () => {
  it('parses records split across arbitrary byte chunks', async () => {
    const events = [];
    for await (const event of readSse(
      byteStream(['data: {"value":', '1}\r\n\r\nevent: note\r\ndata: two', '\r\ndata: lines\r\n\r\n'])
    )) {
      events.push(event);
    }

    expect(events).toEqual([
      { event: 'message', data: '{"value":1}', id: undefined },
      { event: 'note', data: 'two\nlines', id: undefined }
    ]);
  });

  it('ignores comments and emits a final unterminated record', async () => {
    const events = [];
    for await (const event of readSse(byteStream([': keepalive\n\nid: 7\ndata: final']))) {
      events.push(event);
    }
    expect(events).toEqual([{ event: 'message', data: 'final', id: '7' }]);
  });
});
