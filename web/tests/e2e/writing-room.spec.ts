import { expect, test } from '@playwright/test';

test('an author configures a provider, writes, chats, reloads, and exports', async ({ page }) => {
  await page.route('**/v1/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ object: 'list', data: [{ id: 'story-model', owned_by: 'test' }] })
    });
  });
  await page.route('**/v1/chat/completions', async (route) => {
    const request = route.request().postDataJSON();
    expect(request.model).toBe('story-model');
    expect(request.messages.at(-1)).toEqual({ role: 'user', content: 'What is missing?' });
    expect(request.messages.some((message: { content: string }) => message.content.includes('Rain worried the windows.'))).toBeTruthy();
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body:
        'data: {"choices":[{"delta":{"role":"assistant","content":"A sharper desire "}}]}\n\n' +
        'data: {"choices":[{"delta":{"content":"for Mara."}}],"timings":{"predicted_per_second":18.5}}\n\n' +
        'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":6,"total_tokens":46}}\n\n' +
        'data: [DONE]\n\n'
    });
  });

  await page.goto('/');
  await expect(page.getByRole('dialog', { name: 'Language model provider' })).toBeVisible();
  await page.getByRole('button', { name: 'Find models' }).click();
  await expect(page.getByText('Found 1 model.')).toBeVisible();
  await page.getByRole('button', { name: 'Use this provider' }).click();

  await page.getByLabel('Book title').fill('The Glass Orchard');
  await page.getByLabel('Scene title').fill('Weather at the Door');
  await page.getByLabel('Manuscript').fill('Rain worried the windows. Mara waited without lighting the hall.');
  await page.getByPlaceholder('Ask about the story…').fill('What is missing?');
  await page.getByRole('button', { name: 'Send message' }).click();

  await expect(page.getByText('A sharper desire for Mara.')).toBeVisible();
  await expect(page.getByText('18.5 tok/s')).toBeVisible();
  await expect(page.getByText('Saved locally')).toBeVisible();

  await page.reload();
  await expect(page.getByLabel('Book title')).toHaveValue('The Glass Orchard');
  await expect(page.getByLabel('Manuscript')).toHaveValue('Rain worried the windows. Mara waited without lighting the hall.');
  await expect(page.getByText('A sharper desire for Mara.')).toBeVisible();

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('the-glass-orchard.md');
});
