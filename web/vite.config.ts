import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'STRATA_UI_');
  const serverOrigin = env.STRATA_UI_DEV_SERVER_ORIGIN || 'http://127.0.0.1:8080';

  return {
    plugins: [sveltekit()],
    server: {
      proxy: {
        '/v1': {
          target: serverOrigin,
          changeOrigin: true
        }
      }
    },
    test: {
      environment: 'node',
      include: ['src/**/*.test.ts']
    }
  };
});
