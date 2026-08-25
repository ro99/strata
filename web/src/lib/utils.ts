export function createId(prefix: string): string {
  const random = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
  return `${prefix}-${random}`;
}

export function isoNow(): string {
  return new Date().toISOString();
}

export function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/u).length : 0;
}

export function formatRelativeTime(iso: string): string {
  const deltaSeconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (deltaSeconds < 60) return 'now';
  const minutes = Math.floor(deltaSeconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

export function titleFromPrompt(prompt: string): string {
  const collapsed = prompt.replace(/\s+/gu, ' ').trim();
  if (!collapsed) return 'Untitled scene';
  return collapsed.length > 48 ? `${collapsed.slice(0, 47)}…` : collapsed;
}
