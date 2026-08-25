import type { ProviderSettings, WorkspaceSnapshot } from '$lib/types';

const DATABASE_NAME = 'strata-ui';
const DATABASE_VERSION = 1;
const STORE_NAME = 'workspace';
const WORKSPACE_KEY = 'current';
const SETTINGS_KEY = 'strata-ui.provider.v1';

export const defaultProviderSettings: ProviderSettings = {
  baseUrl: '/v1',
  apiKey: '',
  model: '',
  temperature: 0.8,
  maxTokens: 1024,
  systemPrompt:
    'You are a thoughtful creative-writing collaborator. Preserve the author\'s voice and intent. Offer concrete prose, questions, or editorial observations as requested. Never present invented facts about the manuscript as established canon.'
};

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) database.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Could not open browser storage.'));
  });
}

export async function loadWorkspace(): Promise<WorkspaceSnapshot | null> {
  const database = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const request = database
        .transaction(STORE_NAME, 'readonly')
        .objectStore(STORE_NAME)
        .get(WORKSPACE_KEY);
      request.onsuccess = () => resolve((request.result as WorkspaceSnapshot | undefined) ?? null);
      request.onerror = () => reject(request.error ?? new Error('Could not read the writing room.'));
    });
  } finally {
    database.close();
  }
}

export async function saveWorkspace(snapshot: WorkspaceSnapshot): Promise<void> {
  const database = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, 'readwrite');
      transaction.objectStore(STORE_NAME).put(snapshot, WORKSPACE_KEY);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error('Could not save the writing room.'));
    });
  } finally {
    database.close();
  }
}

export function loadProviderSettings(): ProviderSettings {
  try {
    const parsed = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}') as Partial<ProviderSettings>;
    return {
      ...defaultProviderSettings,
      ...parsed,
      temperature: Number.isFinite(parsed.temperature)
        ? Number(parsed.temperature)
        : defaultProviderSettings.temperature,
      maxTokens: Number.isFinite(parsed.maxTokens)
        ? Number(parsed.maxTokens)
        : defaultProviderSettings.maxTokens
    };
  } catch {
    return { ...defaultProviderSettings };
  }
}

export function saveProviderSettings(settings: ProviderSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}
