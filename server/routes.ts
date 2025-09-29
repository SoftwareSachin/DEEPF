import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { createProxyMiddleware } from 'http-proxy-middleware';

export async function registerRoutes(app: Express): Promise<Server> {
  // Proxy API calls to Python backend
  app.use('/api', createProxyMiddleware({
    target: 'http://localhost:8000',
    changeOrigin: true,
    pathRewrite: {
      '^/': '/api/', // Add /api prefix back since Express strips it
    },
    onError: (err: any, req: any, res: any) => {
      console.error('Proxy error:', err);
      res.status(504).json({ error: 'Backend service unavailable' });
    },
    onProxyReq: (proxyReq, req, res) => {
      console.log('Proxy request:', req.method, req.url, '-> http://localhost:8000' + proxyReq.path);
    },
  }));

  // Health check endpoint for Node.js backend
  app.get('/health', (req, res) => {
    res.json({ status: 'Node.js backend healthy', timestamp: new Date().toISOString() });
  });

  const httpServer = createServer(app);

  return httpServer;
}
