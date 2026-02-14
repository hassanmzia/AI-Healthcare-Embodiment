const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = process.env.PORT || 4000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://backend:8000';
const MCP_URL = process.env.MCP_URL || 'http://mcp_server:9000';
const A2A_URL = process.env.A2A_URL || 'http://a2a_gateway:9100';

// Middleware
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({
  origin: ['http://108.48.39.238:3055', 'http://localhost:3055'],
  credentials: true,
}));
app.use(compression());
app.use(morgan('combined'));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000,
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'node-gateway', timestamp: new Date().toISOString() });
});

// Proxy to Django backend API
// Express strips the '/api' mount prefix before passing to the proxy,
// so we prepend '/api' back via pathRewrite to match Django URL config.
app.use('/api', createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  pathRewrite: (path) => `/api${path}`,
  onProxyReq: (proxyReq, req) => {
    proxyReq.setHeader('X-Forwarded-For', req.ip);
  },
  onError: (err, req, res) => {
    console.error('Backend proxy error:', err.message);
    res.status(502).json({ error: 'Backend service unavailable' });
  },
}));

// Proxy to MCP server
app.use('/mcp', createProxyMiddleware({
  target: MCP_URL,
  changeOrigin: true,
  pathRewrite: { '^/mcp': '' },
  onError: (err, req, res) => {
    console.error('MCP proxy error:', err.message);
    res.status(502).json({ error: 'MCP service unavailable' });
  },
}));

// Proxy to A2A gateway
app.use('/a2a', createProxyMiddleware({
  target: A2A_URL,
  changeOrigin: true,
  pathRewrite: { '^/a2a': '' },
  onError: (err, req, res) => {
    console.error('A2A proxy error:', err.message);
    res.status(502).json({ error: 'A2A service unavailable' });
  },
}));

app.listen(PORT, '0.0.0.0', () => {
  console.log(`MS Risk Lab API Gateway running on port ${PORT}`);
  console.log(`  Backend: ${BACKEND_URL}`);
  console.log(`  MCP: ${MCP_URL}`);
  console.log(`  A2A: ${A2A_URL}`);
});
