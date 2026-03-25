#!/usr/bin/env node
import { spawn } from 'child_process';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

// Start Vite dev server
const vite = spawn('npm', ['run', 'dev:renderer'], {
  stdio: 'inherit',
  shell: true,
  cwd: join(__dirname, '..')
});

// Wait for Vite to be ready, then start Electron
const waitOn = require('wait-on');
const electronPath = require('electron');

waitOn({
  resources: ['tcp:5173'],
  timeout: 30000
}).then(() => {
  console.log('\n✓ Vite dev server ready, starting Electron...\n');
  
  const electron = spawn(electronPath, ['.'], {
    stdio: 'inherit',
    cwd: join(__dirname, '..')
  });

  electron.on('close', () => {
    vite.kill();
    process.exit(0);
  });
}).catch(err => {
  console.error('Failed to start dev server:', err);
  vite.kill();
  process.exit(1);
});

process.on('SIGINT', () => {
  vite.kill();
  process.exit(0);
});
