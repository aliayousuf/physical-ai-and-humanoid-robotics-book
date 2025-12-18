const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Proxy middleware for API requests
const apiProxy = createProxyMiddleware('/api/v1', {
  target: process.env.BACKEND_URL || 'http://localhost:8000',
  changeOrigin: true,
  pathRewrite: {
    '^/api/v1': '/api/v1', // Optional: rewrite path
  },
});

// Apply proxy middleware
app.use('/api/v1', apiProxy);

// Serve static files from the build directory
app.use(express.static(path.join(__dirname, 'build')));

// For any route that's not an API call, serve the index.html
app.get('*', (req, res) => {
  // Don't serve index.html for API routes (should be handled by proxy)
  if (req.path.startsWith('/api/')) {
    // This should be handled by the proxy, but as a fallback:
    res.status(404).send('API endpoint not found');
    return;
  }

  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  console.log(`Backend URL: ${process.env.BACKEND_URL || 'http://localhost:8000'}`);
});