const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');
const { app } = require('electron');

class AssetManager {
  constructor() {
    // Get the user data directory for storing downloaded assets
    this.assetsDir = path.join(app.getPath('userData'), 'assets');
    this.modelsDir = path.join(this.assetsDir, 'models');
    this.dataDir = path.join(this.assetsDir, 'data');

    // Ensure directories exist
    this.ensureDirectories();
  }

  ensureDirectories() {
    [this.assetsDir, this.modelsDir, this.dataDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }

  // Download file with progress tracking
  downloadFile(url, destPath, options = {}) {
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(destPath);
      const protocol = url.startsWith('https') ? https : http;

      const request = protocol.get(url, (response) => {
        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download: ${response.statusCode}`));
          return;
        }

        const totalSize = parseInt(response.headers['content-length'], 10);
        let downloadedSize = 0;

        response.on('data', (chunk) => {
          downloadedSize += chunk.length;
          if (options.onProgress && totalSize > 0) {
            const progress = (downloadedSize / totalSize) * 100;
            options.onProgress(progress);
          }
        });

        response.pipe(file);

        file.on('finish', () => {
          file.close();
          console.log(`✅ Downloaded: ${path.basename(destPath)}`);
          resolve(destPath);
        });
      });

      request.on('error', (error) => {
        fs.unlink(destPath, () => {}); // Delete partial file
        reject(error);
      });

      file.on('error', (error) => {
        fs.unlink(destPath, () => {}); // Delete partial file
        reject(error);
      });
    });
  }

  // Check if asset exists locally
  assetExists(filename) {
    const possiblePaths = [
      path.join(this.modelsDir, filename),
      path.join(this.dataDir, filename)
    ];
    return possiblePaths.some(p => fs.existsSync(p));
  }

  // Get local path for asset
  getAssetPath(filename) {
    const possiblePaths = [
      path.join(this.modelsDir, filename),
      path.join(this.dataDir, filename)
    ];

    for (const p of possiblePaths) {
      if (fs.existsSync(p)) {
        return p;
      }
    }
    return null;
  }

  // Download all required assets
  async downloadRequiredAssets() {
    const assets = [
      // ML Models
      {
        url: 'https://example.com/models/chess_model.pkl',
        filename: 'chess_model.pkl',
        type: 'model'
      },
      {
        url: 'https://example.com/models/position_evaluator.bin',
        filename: 'position_evaluator.bin',
        type: 'model'
      },
      // Chess Data
      {
        url: 'https://example.com/data/opening_book.json',
        filename: 'opening_book.json',
        type: 'data'
      },
      {
        url: 'https://example.com/data/endgame_tablebase.zip',
        filename: 'endgame_tablebase.zip',
        type: 'data'
      }
    ];

    const downloads = [];

    for (const asset of assets) {
      if (!this.assetExists(asset.filename)) {
        const destPath = asset.type === 'model'
          ? path.join(this.modelsDir, asset.filename)
          : path.join(this.dataDir, asset.filename);

        console.log(`📥 Downloading ${asset.filename}...`);
        downloads.push(this.downloadFile(asset.url, destPath));
      } else {
        console.log(`✅ Asset already exists: ${asset.filename}`);
      }
    }

    if (downloads.length > 0) {
      await Promise.all(downloads);
      console.log('🎉 All assets downloaded successfully!');
    } else {
      console.log('📋 All assets are already available');
    }

    return true;
  }

  // Get asset information
  getAssetInfo() {
    const models = fs.readdirSync(this.modelsDir).filter(f => !f.startsWith('.'));
    const data = fs.readdirSync(this.dataDir).filter(f => !f.startsWith('.'));

    return {
      modelsDir: this.modelsDir,
      dataDir: this.dataDir,
      models: models.map(f => ({
        name: f,
        path: path.join(this.modelsDir, f),
        size: fs.statSync(path.join(this.modelsDir, f)).size
      })),
      data: data.map(f => ({
        name: f,
        path: path.join(this.dataDir, f),
        size: fs.statSync(path.join(this.dataDir, f)).size
      }))
    };
  }
}

module.exports = AssetManager;
