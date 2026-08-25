export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  status?: 'streaming' | 'complete' | 'error';
  error?: string;
}

export interface Scene {
  id: string;
  title: string;
  content: string;
  updatedAt: string;
}

export interface WritingProject {
  id: string;
  title: string;
  synopsis: string;
  scenes: Scene[];
  activeSceneId: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

export interface WorkspaceSnapshot {
  version: 1;
  projects: WritingProject[];
  activeProjectId: string;
}

export interface ProviderSettings {
  baseUrl: string;
  apiKey: string;
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
}

export interface ModelOption {
  id: string;
  ownedBy?: string;
}

export interface CompletionUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

export interface CompletionTimings {
  promptPerSecond?: number;
  predictedPerSecond?: number;
}

export interface CompletionResult {
  content: string;
  usage?: CompletionUsage;
  timings?: CompletionTimings;
}

export interface StreamCallbacks {
  onText: (text: string) => void;
  onUsage?: (usage: CompletionUsage) => void;
  onTimings?: (timings: CompletionTimings) => void;
}
