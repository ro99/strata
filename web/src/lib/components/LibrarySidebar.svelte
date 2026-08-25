<script lang="ts">
  import type { WritingProject } from '$lib/types';
  import { countWords, formatRelativeTime } from '$lib/utils';

  interface Props {
    projects: WritingProject[];
    activeProjectId: string;
    saveState: 'idle' | 'saving' | 'saved' | 'error';
    saveError: string;
    onSelectProject: (id: string) => void;
    onAddProject: () => void;
    onSelectScene: (id: string) => void;
    onAddScene: () => void;
    onOpenSettings: () => void;
  }

  let {
    projects,
    activeProjectId,
    saveState,
    saveError,
    onSelectProject,
    onAddProject,
    onSelectScene,
    onAddScene,
    onOpenSettings
  }: Props = $props();

  let activeProject = $derived(projects.find((project) => project.id === activeProjectId));
</script>

<aside class="library-panel">
  <header class="brand-block">
    <div class="brand-mark" aria-hidden="true">S</div>
    <div>
      <p class="eyebrow">Strata</p>
      <h1>Writing Room</h1>
    </div>
  </header>

  <div class="library-scroll">
    <section class="library-section" aria-labelledby="books-heading">
      <div class="section-heading">
        <h2 id="books-heading">Books</h2>
        <button class="icon-button" type="button" onclick={onAddProject} title="New book" aria-label="New book">+</button>
      </div>
      <nav class="project-list" aria-label="Books">
        {#each projects as project (project.id)}
          <button
            type="button"
            class:active={project.id === activeProjectId}
            onclick={() => onSelectProject(project.id)}
          >
            <span>{project.title || 'Untitled novel'}</span>
            <small>{project.scenes.length} {project.scenes.length === 1 ? 'scene' : 'scenes'}</small>
          </button>
        {/each}
      </nav>
    </section>

    {#if activeProject}
      <section class="library-section" aria-labelledby="scenes-heading">
        <div class="section-heading">
          <h2 id="scenes-heading">Scenes</h2>
          <button class="icon-button" type="button" onclick={onAddScene} title="New scene" aria-label="New scene">+</button>
        </div>
        <nav class="scene-list" aria-label="Scenes">
          {#each activeProject.scenes as scene, index (scene.id)}
            <button
              type="button"
              class:active={scene.id === activeProject.activeSceneId}
              onclick={() => onSelectScene(scene.id)}
            >
              <span class="scene-number">{String(index + 1).padStart(2, '0')}</span>
              <span class="scene-copy">
                <strong>{scene.title || 'Untitled scene'}</strong>
                <small>{countWords(scene.content)} words · {formatRelativeTime(scene.updatedAt)}</small>
              </span>
            </button>
          {/each}
        </nav>
      </section>
    {/if}
  </div>

  <footer class="library-footer">
    <div class="save-indicator" class:error={saveState === 'error'} title={saveError}>
      <span class="save-dot"></span>
      {saveState === 'saving'
        ? 'Saving…'
        : saveState === 'error'
          ? 'Not saved'
          : 'Saved locally'}
    </div>
    <button class="settings-button" type="button" onclick={onOpenSettings}>
      <span aria-hidden="true">⚙</span>
      Provider
    </button>
  </footer>
</aside>
