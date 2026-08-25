<script lang="ts">
  import type { ChatMessage } from '$lib/types';

  interface Props {
    messages: ChatMessage[];
    prompt: string;
    generating: boolean;
    includeScene: boolean;
    modelLabel: string;
    metricsLabel: string;
    onPrompt: (value: string) => void;
    onIncludeScene: (value: boolean) => void;
    onSend: () => void;
    onStop: () => void;
    onClear: () => void;
    onOpenSettings: () => void;
  }

  let {
    messages,
    prompt,
    generating,
    includeScene,
    modelLabel,
    metricsLabel,
    onPrompt,
    onIncludeScene,
    onSend,
    onStop,
    onClear,
    onOpenSettings
  }: Props = $props();

  let transcript: HTMLDivElement;
  let composer: HTMLTextAreaElement;
  let messageSignature = $derived(messages.map((message) => message.content.length).join(':'));

  $effect(() => {
    messageSignature;
    requestAnimationFrame(() => transcript?.scrollTo({ top: transcript.scrollHeight, behavior: 'smooth' }));
  });

  const suggestions = [
    'What possibilities do you see for the next beat?',
    'Give me an honest developmental edit of this scene.',
    'Help me deepen the emotional subtext without explaining it.',
    'Suggest five complications that fit the established tone.'
  ];

  function keydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!generating && prompt.trim()) onSend();
    }
  }

  function chooseSuggestion(value: string): void {
    onPrompt(value);
    requestAnimationFrame(() => composer?.focus());
  }
</script>

<aside class="assistant-panel">
  <header class="assistant-header">
    <div>
      <p class="eyebrow">Creative collaborator</p>
      <h2>Story conversation</h2>
    </div>
    <div class="assistant-actions">
      <button class="icon-button" type="button" onclick={onClear} title="Clear conversation" aria-label="Clear conversation">↺</button>
      <button class="icon-button" type="button" onclick={onOpenSettings} title="Provider settings" aria-label="Provider settings">⋯</button>
    </div>
  </header>

  <div class="provider-strip">
    <span class:offline={modelLabel === 'Choose a model'}></span>
    <button type="button" onclick={onOpenSettings}>{modelLabel}</button>
    {#if metricsLabel}<small>{metricsLabel}</small>{/if}
  </div>

  <div class="transcript" bind:this={transcript} aria-live="polite">
    {#if messages.length === 0}
      <section class="assistant-welcome">
        <div class="quill" aria-hidden="true">✦</div>
        <h3>Stay in the work.</h3>
        <p>Ask for possibilities, critique, research questions, or a second pair of eyes. Your draft remains in this browser.</p>
        <div class="suggestion-list">
          {#each suggestions as suggestion}
            <button type="button" onclick={() => chooseSuggestion(suggestion)}>{suggestion}</button>
          {/each}
        </div>
      </section>
    {:else}
      {#each messages as message (message.id)}
        <article class="chat-message" class:user={message.role === 'user'} class:error={message.status === 'error'}>
          <header>{message.role === 'user' ? 'You' : 'Collaborator'}</header>
          <div>{message.content || (message.status === 'streaming' ? 'Thinking…' : '')}</div>
          {#if message.error}<small>{message.error}</small>{/if}
        </article>
      {/each}
    {/if}
  </div>

  <footer class="composer-block">
    <label class="context-toggle">
      <input
        type="checkbox"
        checked={includeScene}
        onchange={(event) => onIncludeScene(event.currentTarget.checked)}
      />
      <span>Share current scene as context</span>
    </label>
    <form
      class="composer"
      onsubmit={(event) => {
        event.preventDefault();
        if (!generating && prompt.trim()) onSend();
      }}
    >
      <textarea
        bind:this={composer}
        value={prompt}
        oninput={(event) => onPrompt(event.currentTarget.value)}
        onkeydown={keydown}
        placeholder="Ask about the story…"
        rows="3"
      ></textarea>
      {#if generating}
        <button class="send-button stop" type="button" onclick={onStop} aria-label="Stop generation">■</button>
      {:else}
        <button class="send-button" type="submit" disabled={!prompt.trim()} aria-label="Send message">↑</button>
      {/if}
    </form>
    <p class="composer-hint">Enter to send · Shift+Enter for a new line</p>
  </footer>
</aside>
