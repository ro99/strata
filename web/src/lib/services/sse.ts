export interface SseEvent {
  event: string;
  data: string;
  id?: string;
}

function parseRecord(record: string): SseEvent | null {
  let event = 'message';
  let id: string | undefined;
  const data: string[] = [];

  for (const line of record.split(/\r?\n/u)) {
    if (!line || line.startsWith(':')) continue;
    const separator = line.indexOf(':');
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? '' : line.slice(separator + 1);
    if (value.startsWith(' ')) value = value.slice(1);

    if (field === 'data') data.push(value);
    else if (field === 'event') event = value;
    else if (field === 'id') id = value;
  }

  if (data.length === 0) return null;
  return { event, data: data.join('\n'), id };
}

export async function* readSse(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal
): AsyncGenerator<SseEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (!signal?.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/gu, '\n');

      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const parsed = parseRecord(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (parsed) yield parsed;
        boundary = buffer.indexOf('\n\n');
      }
    }

    buffer += decoder.decode().replace(/\r\n/gu, '\n');
    if (buffer.trim()) {
      const parsed = parseRecord(buffer);
      if (parsed) yield parsed;
    }
  } finally {
    reader.releaseLock();
  }
}
