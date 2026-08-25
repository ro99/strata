import { describe, expect, it } from 'vitest';
import { countWords, titleFromPrompt } from './utils';

describe('writing utilities', () => {
  it('counts Unicode text separated by whitespace', () => {
    expect(countWords('  One\n two\ttrês  ')).toBe(3);
    expect(countWords('')).toBe(0);
  });

  it('creates bounded scene labels from prose', () => {
    expect(titleFromPrompt('  A   quiet opening  ')).toBe('A quiet opening');
    expect(titleFromPrompt('x'.repeat(60))).toHaveLength(48);
  });
});
