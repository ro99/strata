<script lang="ts">
  import type { Scene, WritingProject } from '$lib/types';
  import { countWords } from '$lib/utils';

  interface Props {
    project: WritingProject;
    scene: Scene;
    onProjectTitle: (value: string) => void;
    onSynopsis: (value: string) => void;
    onSceneTitle: (value: string) => void;
    onSceneContent: (value: string) => void;
    onExport: () => void;
  }

  let {
    project,
    scene,
    onProjectTitle,
    onSynopsis,
    onSceneTitle,
    onSceneContent,
    onExport
  }: Props = $props();

  let words = $derived(countWords(scene.content));
  let characters = $derived(scene.content.length);
</script>

<main class="manuscript-panel">
  <header class="manuscript-header">
    <label class="book-title-field">
      <span class="sr-only">Book title</span>
      <input
        value={project.title}
        oninput={(event) => onProjectTitle(event.currentTarget.value)}
        placeholder="Untitled novel"
      />
    </label>
    <button class="quiet-button" type="button" onclick={onExport}>
      <span aria-hidden="true">↓</span> Export
    </button>
  </header>

  <label class="book-note-field">
    <span>Book compass</span>
    <input
      value={project.synopsis}
      oninput={(event) => onSynopsis(event.currentTarget.value)}
      placeholder="A one-line promise, premise, or north star for this book"
    />
  </label>

  <article class="paper">
    <label class="scene-title-field">
      <span class="sr-only">Scene title</span>
      <input
        value={scene.title}
        oninput={(event) => onSceneTitle(event.currentTarget.value)}
        placeholder="Scene title"
      />
    </label>
    <div class="ornament" aria-hidden="true"><span></span><b>◆</b><span></span></div>
    <label class="draft-field">
      <span class="sr-only">Manuscript</span>
      <textarea
        value={scene.content}
        oninput={(event) => onSceneContent(event.currentTarget.value)}
        placeholder="Begin the scene…"
        spellcheck="true"
      ></textarea>
    </label>
    <footer class="manuscript-stats" aria-live="polite">
      <span>{words.toLocaleString()} words</span>
      <span>{characters.toLocaleString()} characters</span>
    </footer>
  </article>
</main>
