<script lang="ts">
  import { onMount } from 'svelte';
  import LibrarySidebar from '$lib/components/LibrarySidebar.svelte';
  import ManuscriptEditor from '$lib/components/ManuscriptEditor.svelte';
  import ProviderDialog from '$lib/components/ProviderDialog.svelte';
  import WritingAssistant from '$lib/components/WritingAssistant.svelte';
  import { listModels, streamCompletion } from '$lib/services/openai';
  import { workspace } from '$lib/state/workspace.svelte';
  import type { ChatMessage, CompletionTimings, CompletionUsage, ProviderSettings } from '$lib/types';
  import { createId, isoNow } from '$lib/utils';

  let prompt = $state('');
  let includeScene = $state(true);
  let settingsOpen = $state(false);
  let providerLoading = $state(false);
  let providerError = $state('');
  let generating = $state(false);
  let usage = $state<CompletionUsage | undefined>();
  let timings = $state<CompletionTimings | undefined>();
  let abortController: AbortController | undefined;

  let project = $derived(workspace.activeProject);
  let scene = $derived(workspace.activeScene);
  let modelLabel = $derived(workspace.provider.model || 'Choose a model');
  let metricsLabel = $derived.by(() => {
    const speed = timings?.predictedPerSecond;
    if (typeof speed === 'number' && Number.isFinite(speed)) return `${speed.toFixed(1)} tok/s`;
    const tokens = usage?.completionTokens;
    if (typeof tokens === 'number') return `${tokens} tokens`;
    return '';
  });

  onMount(() => {
    let active = true;
    void workspace.initialize().then(() => {
      if (active && !workspace.provider.model) settingsOpen = true;
    });
    const beforeUnload = () => void workspace.flushSave();
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      active = false;
      window.removeEventListener('beforeunload', beforeUnload);
    };
  });

  async function discoverModels(settings: ProviderSettings): Promise<void> {
    providerLoading = true;
    providerError = '';
    try {
      const models = await listModels(settings);
      workspace.setModels(models);
      if (!settings.model && models.length === 1) settings.model = models[0].id;
    } catch (error) {
      providerError = error instanceof Error ? error.message : 'Could not connect to the provider.';
      workspace.setModels([]);
    } finally {
      providerLoading = false;
    }
  }

  function saveProvider(settings: ProviderSettings): void {
    if (!settings.model.trim()) {
      providerError = 'Choose a model or enter its ID.';
      return;
    }
    workspace.setProvider(settings);
    providerError = '';
    settingsOpen = false;
  }

  async function sendMessage(): Promise<void> {
    const activeProject = project;
    const activeScene = scene;
    const text = prompt.trim();
    if (!activeProject || !activeScene || !text || generating) return;
    if (!workspace.provider.model.trim()) {
      providerError = 'Configure a provider and model before starting the conversation.';
      settingsOpen = true;
      return;
    }

    const projectId = activeProject.id;
    const history: ChatMessage[] = activeProject.messages.filter(
      (message) => message.status !== 'error' && message.status !== 'streaming'
    );
    const userMessage: ChatMessage = {
      id: createId('message'),
      role: 'user',
      content: text,
      createdAt: isoNow(),
      status: 'complete'
    };
    const assistantMessage: ChatMessage = {
      id: createId('message'),
      role: 'assistant',
      content: '',
      createdAt: isoNow(),
      status: 'streaming'
    };

    workspace.appendMessage(userMessage, projectId);
    workspace.appendMessage(assistantMessage, projectId);
    prompt = '';
    usage = undefined;
    timings = undefined;
    generating = true;
    abortController = new AbortController();

    let pendingText = '';
    let receivedText = '';
    let animationFrame: number | undefined;
    const flushText = () => {
      if (animationFrame !== undefined) cancelAnimationFrame(animationFrame);
      animationFrame = undefined;
      if (!pendingText) return;
      workspace.appendToMessage(assistantMessage.id, pendingText, projectId);
      pendingText = '';
    };
    const scheduleText = (value: string) => {
      receivedText += value;
      pendingText += value;
      if (animationFrame === undefined) animationFrame = requestAnimationFrame(flushText);
    };

    try {
      await streamCompletion({
        settings: workspace.provider,
        messages: [...history, userMessage],
        sceneContext: includeScene
          ? [activeProject.synopsis && `Book compass: ${activeProject.synopsis}`, activeScene.content]
              .filter(Boolean)
              .join('\n\n')
          : undefined,
        signal: abortController.signal,
        callbacks: {
          onText: scheduleText,
          onUsage: (nextUsage) => (usage = nextUsage),
          onTimings: (nextTimings) => (timings = nextTimings)
        }
      });
      flushText();
      workspace.finishMessage(assistantMessage.id, projectId);
    } catch (error) {
      flushText();
      if (error instanceof DOMException && error.name === 'AbortError') {
        if (receivedText) workspace.finishMessage(assistantMessage.id, projectId);
        else workspace.removeMessage(assistantMessage.id, projectId);
      } else {
        const message = error instanceof Error ? error.message : 'The provider connection failed.';
        workspace.failMessage(assistantMessage.id, message, projectId);
      }
    } finally {
      if (animationFrame !== undefined) cancelAnimationFrame(animationFrame);
      abortController = undefined;
      generating = false;
      await workspace.flushSave();
    }
  }

  function stopGeneration(): void {
    abortController?.abort();
  }

  function clearConversation(): void {
    if (generating) return;
    if (project?.messages.length && window.confirm('Clear this book\'s entire model conversation?')) {
      workspace.clearConversation();
      usage = undefined;
      timings = undefined;
    }
  }

  function exportProject(): void {
    if (!project) return;
    const body = [
      `# ${project.title || 'Untitled novel'}`,
      project.synopsis ? `> ${project.synopsis}` : '',
      ...project.scenes.flatMap((item) => [`## ${item.title || 'Untitled scene'}`, item.content])
    ]
      .filter(Boolean)
      .join('\n\n');
    const blob = new Blob([`${body}\n`], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${(project.title || 'untitled-novel').replace(/[^a-z0-9]+/giu, '-').replace(/^-|-$/gu, '').toLowerCase() || 'novel'}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  }
</script>

<svelte:head>
  <title>Strata Writing Room</title>
</svelte:head>

{#if !workspace.hydrated}
  <main class="loading-screen">
    <div class="brand-mark">S</div>
    <p>Opening your writing room…</p>
  </main>
{:else if project && scene}
  <div class="app-shell">
    <LibrarySidebar
      projects={workspace.projects}
      activeProjectId={workspace.activeProjectId}
      saveState={workspace.saveState}
      saveError={workspace.saveError}
      onSelectProject={(id) => workspace.selectProject(id)}
      onAddProject={() => workspace.addProject()}
      onSelectScene={(id) => workspace.selectScene(id)}
      onAddScene={() => workspace.addScene()}
      onOpenSettings={() => (settingsOpen = true)}
    />
    <ManuscriptEditor
      {project}
      {scene}
      onProjectTitle={(value) => workspace.updateProjectTitle(value)}
      onSynopsis={(value) => workspace.updateSynopsis(value)}
      onSceneTitle={(value) => workspace.updateSceneTitle(value)}
      onSceneContent={(value) => workspace.updateSceneContent(value)}
      onExport={exportProject}
    />
    <WritingAssistant
      messages={project.messages}
      {prompt}
      {generating}
      {includeScene}
      {modelLabel}
      {metricsLabel}
      onPrompt={(value) => (prompt = value)}
      onIncludeScene={(value) => (includeScene = value)}
      onSend={() => void sendMessage()}
      onStop={stopGeneration}
      onClear={clearConversation}
      onOpenSettings={() => (settingsOpen = true)}
    />
  </div>
{/if}

<ProviderDialog
  open={settingsOpen}
  settings={workspace.provider}
  models={workspace.models}
  loading={providerLoading}
  error={providerError}
  onClose={() => (settingsOpen = false)}
  onDiscover={(settings) => void discoverModels(settings)}
  onSave={saveProvider}
/>
