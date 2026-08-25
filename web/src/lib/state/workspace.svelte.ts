import type {
  ChatMessage,
  ModelOption,
  ProviderSettings,
  Scene,
  WorkspaceSnapshot,
  WritingProject
} from '$lib/types';
import { createId, isoNow } from '$lib/utils';
import {
  defaultProviderSettings,
  loadProviderSettings,
  loadWorkspace,
  saveProviderSettings,
  saveWorkspace
} from '$lib/services/storage';

function newScene(title = 'Opening scene'): Scene {
  return { id: createId('scene'), title, content: '', updatedAt: isoNow() };
}

function newProject(title = 'Untitled novel'): WritingProject {
  const scene = newScene();
  const now = isoNow();
  return {
    id: createId('project'),
    title,
    synopsis: '',
    scenes: [scene],
    activeSceneId: scene.id,
    messages: [],
    createdAt: now,
    updatedAt: now
  };
}

function freshSnapshot(): WorkspaceSnapshot {
  const project = newProject();
  return { version: 1, projects: [project], activeProjectId: project.id };
}

function validSnapshot(value: WorkspaceSnapshot | null): value is WorkspaceSnapshot {
  return Boolean(
    value &&
      value.version === 1 &&
      Array.isArray(value.projects) &&
      value.projects.length > 0 &&
      value.projects.some((project) => project.id === value.activeProjectId)
  );
}

export class WorkspaceState {
  projects = $state<WritingProject[]>([]);
  activeProjectId = $state('');
  provider = $state<ProviderSettings>({ ...defaultProviderSettings });
  models = $state<ModelOption[]>([]);
  hydrated = $state(false);
  saveState = $state<'idle' | 'saving' | 'saved' | 'error'>('idle');
  saveError = $state('');
  private saveTimer: ReturnType<typeof setTimeout> | undefined;

  get activeProject(): WritingProject | undefined {
    return this.projects.find((project) => project.id === this.activeProjectId);
  }

  get activeScene(): Scene | undefined {
    const project = this.activeProject;
    return project?.scenes.find((scene) => scene.id === project.activeSceneId);
  }

  async initialize(): Promise<void> {
    if (this.hydrated) return;
    this.provider = loadProviderSettings();
    try {
      const stored = await loadWorkspace();
      const snapshot = validSnapshot(stored) ? stored : freshSnapshot();
      this.projects = snapshot.projects;
      this.activeProjectId = snapshot.activeProjectId;
    } catch (error) {
      const snapshot = freshSnapshot();
      this.projects = snapshot.projects;
      this.activeProjectId = snapshot.activeProjectId;
      this.saveState = 'error';
      this.saveError = error instanceof Error ? error.message : 'Browser storage is unavailable.';
    } finally {
      this.hydrated = true;
    }
  }

  setProvider(settings: ProviderSettings): void {
    this.provider = { ...settings };
    saveProviderSettings(this.provider);
  }

  setModels(models: ModelOption[]): void {
    this.models = models;
  }

  addProject(): WritingProject {
    const project = newProject();
    this.projects.push(project);
    this.activeProjectId = project.id;
    this.queueSave();
    return project;
  }

  selectProject(projectId: string): void {
    if (this.projects.some((project) => project.id === projectId)) {
      this.activeProjectId = projectId;
      this.queueSave();
    }
  }

  updateProjectTitle(title: string): void {
    const project = this.activeProject;
    if (!project) return;
    project.title = title;
    this.touch(project);
  }

  updateSynopsis(synopsis: string): void {
    const project = this.activeProject;
    if (!project) return;
    project.synopsis = synopsis;
    this.touch(project);
  }

  addScene(): Scene | undefined {
    const project = this.activeProject;
    if (!project) return undefined;
    const scene = newScene(`Scene ${project.scenes.length + 1}`);
    project.scenes.push(scene);
    project.activeSceneId = scene.id;
    this.touch(project);
    return scene;
  }

  selectScene(sceneId: string): void {
    const project = this.activeProject;
    if (!project?.scenes.some((scene) => scene.id === sceneId)) return;
    project.activeSceneId = sceneId;
    this.touch(project);
  }

  updateSceneTitle(title: string): void {
    const project = this.activeProject;
    const scene = this.activeScene;
    if (!project || !scene) return;
    scene.title = title;
    scene.updatedAt = isoNow();
    this.touch(project);
  }

  updateSceneContent(content: string): void {
    const project = this.activeProject;
    const scene = this.activeScene;
    if (!project || !scene) return;
    scene.content = content;
    scene.updatedAt = isoNow();
    this.touch(project, 450);
  }

  appendMessage(message: ChatMessage, projectId = this.activeProjectId): void {
    const project = this.projects.find((candidate) => candidate.id === projectId);
    if (!project) return;
    project.messages.push(message);
    this.touch(project);
  }

  appendToMessage(messageId: string, text: string, projectId = this.activeProjectId): void {
    const project = this.projects.find((candidate) => candidate.id === projectId);
    const message = project?.messages.find((candidate) => candidate.id === messageId);
    if (!project || !message) return;
    message.content += text;
    this.touch(project, 200);
  }

  finishMessage(messageId: string, projectId = this.activeProjectId): void {
    const project = this.projects.find((candidate) => candidate.id === projectId);
    const message = project?.messages.find((candidate) => candidate.id === messageId);
    if (!project || !message) return;
    message.status = 'complete';
    this.touch(project, 0);
  }

  failMessage(messageId: string, error: string, projectId = this.activeProjectId): void {
    const project = this.projects.find((candidate) => candidate.id === projectId);
    const message = project?.messages.find((candidate) => candidate.id === messageId);
    if (!project || !message) return;
    message.status = 'error';
    message.error = error;
    this.touch(project, 0);
  }

  removeMessage(messageId: string, projectId = this.activeProjectId): void {
    const project = this.projects.find((candidate) => candidate.id === projectId);
    if (!project) return;
    project.messages = project.messages.filter((message) => message.id !== messageId);
    this.touch(project, 0);
  }

  clearConversation(): void {
    const project = this.activeProject;
    if (!project) return;
    project.messages = [];
    this.touch(project, 0);
  }

  snapshot(): WorkspaceSnapshot {
    return {
      version: 1,
      projects: $state.snapshot(this.projects),
      activeProjectId: this.activeProjectId
    };
  }

  async flushSave(): Promise<void> {
    if (!this.hydrated) return;
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = undefined;
    this.saveState = 'saving';
    this.saveError = '';
    try {
      await saveWorkspace(this.snapshot());
      this.saveState = 'saved';
    } catch (error) {
      this.saveState = 'error';
      this.saveError = error instanceof Error ? error.message : 'Could not save changes.';
    }
  }

  private touch(project: WritingProject, delay = 150): void {
    project.updatedAt = isoNow();
    this.queueSave(delay);
  }

  private queueSave(delay = 150): void {
    if (!this.hydrated) return;
    this.saveState = 'idle';
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => void this.flushSave(), delay);
  }
}

export const workspace = new WorkspaceState();
