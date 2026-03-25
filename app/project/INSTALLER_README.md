# Chess Analyser - Desktop App Installer Guide

## 🚀 Building the Installer

To create a downloadable installer for your Chess Analyser desktop app:

### Prerequisites
- Node.js (v16 or higher)
- Python 3.x (for chess analysis)
- Git

### Build Steps

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Build the renderer:**
   ```bash
   npm run build:renderer
   ```

3. **Create the installer:**
   ```bash
   npm run build:installer
   ```

The installer will be created in the `release/` directory as a `.exe` file (Windows NSIS installer).

## 📦 What's Included in the Installer

The installer contains:
- ✅ Chess Analyser desktop application
- ✅ All necessary UI components
- ✅ Electron runtime
- ❌ Large ML models and chess data files (downloaded separately)

## 🔧 Large Files Management

The app is designed to download large assets (ML models, chess databases) after installation:

### Automatic Download
When users first run the app, it will automatically:
1. Check for required assets
2. Download missing ML models and data files
3. Store them in the user's app data directory

### Manual Management
Users can also manually manage assets through the app's settings.

## 🎯 Features

- **Lightweight Installer**: ~50-100MB (without large assets)
- **Automatic Asset Download**: ML models downloaded on-demand
- **Offline-First**: Core app works without internet after initial setup
- **Windows Compatible**: NSIS installer for easy distribution

## 📋 Distribution

1. **Share the `.exe` installer** with users
2. **Users run the installer** - installs to Program Files
3. **First launch downloads assets** - ~200-500MB of ML models and data
4. **App is ready to use**

## 🛠️ Development

### Project Structure
```
project/
├── src/                 # React frontend
├── electron/           # Electron main/preload scripts
│   ├── main.cjs       # Main process with asset management
│   ├── preload.cjs    # Preload script
│   └── asset-manager.js # Asset download manager
├── assets/            # Icons and resources
├── scripts/           # Build scripts
└── chess-analyzer-backend.js # Express server (optional)
```

### Adding New Assets

To add new large files that should be downloaded separately:

1. Update `electron/asset-manager.js`
2. Add asset URLs and metadata
3. The app will automatically handle downloads

## 🔍 Troubleshooting

### Build Issues
- Ensure all dependencies are installed: `npm install`
- Check that Python is available in PATH
- Verify Node.js version compatibility

### Runtime Issues
- Assets are stored in `%APPDATA%/Chess Analyser/assets/`
- Check app logs for download errors
- Internet connection required for first-time asset download

## 📊 File Sizes

- **Base Installer**: ~50-100MB
- **ML Models**: ~200-300MB (downloaded)
- **Chess Data**: ~100-200MB (downloaded)
- **Total After Setup**: ~350-600MB

This approach keeps your installer small while providing full functionality!
