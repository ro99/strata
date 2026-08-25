<script lang="ts">
  import type { ModelOption, ProviderSettings } from '$lib/types';

  interface Props {
    open: boolean;
    settings: ProviderSettings;
    models: ModelOption[];
    loading: boolean;
    error: string;
    onClose: () => void;
    onDiscover: (settings: ProviderSettings) => void;
    onSave: (settings: ProviderSettings) => void;
  }

  let { open, settings, models, loading, error, onClose, onDiscover, onSave }: Props = $props();
  let draft = $state<ProviderSettings>({
    baseUrl: '',
    apiKey: '',
    model: '',
    temperature: 0.8,
    maxTokens: 1024,
    systemPrompt: ''
  });
  let visibleKey = $state(false);

  $effect(() => {
    if (open) draft = { ...settings };
  });

  $effect(() => {
    if (open && models.length === 1 && !draft.model) draft.model = models[0].id;
  });

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    onSave({
      ...draft,
      baseUrl: draft.baseUrl.trim(),
      apiKey: draft.apiKey.trim(),
      model: draft.model.trim(),
      temperature: Number(draft.temperature),
      maxTokens: Number(draft.maxTokens)
    });
  }
</script>

<svelte:window onkeydown={(event) => event.key === 'Escape' && open && onClose()} />

{#if open}
  <div class="dialog-backdrop" role="presentation" onclick={(event) => event.target === event.currentTarget && onClose()}>
    <div class="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="provider-title">
      <header>
        <div>
          <p class="eyebrow">Connection</p>
          <h2 id="provider-title">Language model provider</h2>
        </div>
        <button class="icon-button" type="button" onclick={onClose} aria-label="Close settings">×</button>
      </header>

      <form onsubmit={submit}>
        <div class="settings-intro">
          <strong>Bring your own endpoint.</strong>
          <p>Strata UI speaks the OpenAI-compatible Models and Chat Completions APIs. No provider SDK or Strata-only route is required.</p>
        </div>

        <label class="field-label">
          <span>API base URL</span>
          <input bind:value={draft.baseUrl} placeholder="http://127.0.0.1:8080/v1" required />
        </label>
        <div class="preset-row" aria-label="Endpoint presets">
          <button type="button" onclick={() => (draft.baseUrl = '/v1')}>Same-origin /v1</button>
          <button type="button" onclick={() => (draft.baseUrl = 'http://127.0.0.1:8080/v1')}>Local port 8080</button>
          <button type="button" onclick={() => (draft.baseUrl = 'https://api.openai.com/v1')}>OpenAI</button>
        </div>

        <label class="field-label">
          <span>API key <small>optional for local servers</small></span>
          <span class="secret-field">
            <input bind:value={draft.apiKey} type={visibleKey ? 'text' : 'password'} autocomplete="off" placeholder="sk-…" />
            <button type="button" onclick={() => (visibleKey = !visibleKey)}>{visibleKey ? 'Hide' : 'Show'}</button>
          </span>
        </label>

        <div class="model-field-row">
          <label class="field-label">
            <span>Model</span>
            <input bind:value={draft.model} list="provider-models" placeholder="Model ID" required />
            <datalist id="provider-models">
              {#each models as model}<option value={model.id}>{model.ownedBy ?? ''}</option>{/each}
            </datalist>
          </label>
          <button class="discover-button" type="button" disabled={loading || !draft.baseUrl.trim()} onclick={() => onDiscover({ ...draft })}>
            {loading ? 'Connecting…' : 'Find models'}
          </button>
        </div>
        {#if error}<p class="settings-error" role="alert">{error}</p>{/if}
        {#if models.length > 0}<p class="settings-success">Found {models.length} {models.length === 1 ? 'model' : 'models'}.</p>{/if}

        <div class="settings-grid">
          <label class="field-label">
            <span>Temperature</span>
            <input bind:value={draft.temperature} type="number" min="0" max="2" step="0.05" required />
          </label>
          <label class="field-label">
            <span>Maximum response tokens</span>
            <input bind:value={draft.maxTokens} type="number" min="1" max="131072" step="1" required />
          </label>
        </div>

        <label class="field-label">
          <span>Collaborator instructions</span>
          <textarea bind:value={draft.systemPrompt} rows="5"></textarea>
        </label>

        <p class="privacy-note">
          The endpoint, model, and key are stored only in this browser. A browser-held cloud key is accessible to anyone with access to this browser profile; prefer a restricted key or a trusted local proxy.
        </p>

        <footer>
          <button class="quiet-button" type="button" onclick={onClose}>Cancel</button>
          <button class="primary-button" type="submit">Use this provider</button>
        </footer>
      </form>
    </div>
  </div>
{/if}
