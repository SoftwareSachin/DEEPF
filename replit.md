# Overview

This is a deepfake detection web application that uses AI-powered analysis to identify synthetic media content in images and videos. The system provides a modern web interface for users to upload media files and receive detailed analysis results with confidence scores, face detection, and frame-by-frame video analysis.

The application combines a React frontend with a Python backend that implements lightweight CNN models for deepfake detection, using OpenCV and MediaPipe for face detection and extraction. The system is designed to be accessible and user-friendly while providing professional-grade analysis capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The frontend is built with React 18 and TypeScript, using Vite as the build tool. The application follows a component-based architecture with:

- **UI Framework**: Shadcn/ui components with Radix UI primitives for accessible, customizable components
- **Styling**: Tailwind CSS with a dark-mode-first design system following security tool aesthetics
- **State Management**: React Query (TanStack Query) for server state management and caching
- **Routing**: Wouter for lightweight client-side routing
- **Real-time Communication**: Socket.IO client for live processing updates

The design system emphasizes a professional security tool appearance with generous whitespace, clear visual hierarchy, and color-coded results (green for authentic, red for fake content).

## Backend Architecture
The system uses a dual-backend approach:

- **Node.js/Express Backend**: Handles main application routes, user management, and database operations
- **Python Backend**: Dedicated microservice for deepfake detection and media processing

### Python Detection Service
- **Framework**: Flask with Flask-SocketIO for real-time WebSocket communication
- **AI Models**: Lightweight CNN ensemble implemented in pure NumPy to avoid heavy dependencies
- **Face Detection**: MediaPipe for efficient face detection and extraction
- **Video Processing**: OpenCV for frame extraction and video analysis
- **Processing Pipeline**: Multi-step analysis with real-time progress updates

The Python backend implements a custom lightweight CNN architecture to maintain disk space efficiency while providing reliable deepfake detection capabilities.

## Data Storage Solutions
- **Database**: PostgreSQL with Drizzle ORM for type-safe database operations
- **Connection**: Neon Database serverless PostgreSQL for cloud deployment
- **File Storage**: Local filesystem for uploaded media files and analysis results
- **Session Management**: PostgreSQL-backed sessions using connect-pg-simple

## Authentication and Authorization
The system includes a basic user management schema with username/password authentication, though the current implementation focuses primarily on the core detection functionality.

## Processing Workflow
1. **File Upload**: Frontend handles drag-and-drop and file selection with validation
2. **Real-time Processing**: WebSocket connection provides live updates during analysis
3. **Face Detection**: MediaPipe extracts faces from images/video frames
4. **Deepfake Analysis**: CNN ensemble analyzes extracted faces
5. **Results Display**: Comprehensive results with confidence scores, frame analysis, and downloadable reports

# External Dependencies

## Core Frameworks
- **React 18**: Frontend framework with hooks and modern patterns
- **Express.js**: Node.js web application framework
- **Flask**: Python web framework for AI processing service

## AI and Computer Vision
- **MediaPipe**: Google's face detection and pose estimation
- **OpenCV (cv2)**: Computer vision library for image/video processing
- **NumPy**: Numerical computing for custom CNN implementation
- **scikit-learn**: Machine learning utilities and preprocessing

## Database and Storage
- **PostgreSQL**: Primary database via Neon Database serverless
- **Drizzle ORM**: Type-safe database operations with schema validation

## UI and Styling
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Headless UI components for accessibility
- **Shadcn/ui**: Pre-built component library
- **Lucide React**: Icon library for consistent iconography

## Development Tools
- **TypeScript**: Static type checking for JavaScript
- **Vite**: Fast build tool and development server
- **ESBuild**: Fast JavaScript bundler for production builds

## Real-time Communication
- **Socket.IO**: WebSocket library for real-time client-server communication
- **Flask-SocketIO**: Python WebSocket integration for processing updates

## Font and Asset Delivery
- **Google Fonts**: Inter and JetBrains Mono fonts via CDN
- **PostCSS**: CSS processing with Autoprefixer

The system is designed to be lightweight yet powerful, avoiding heavy ML frameworks like TensorFlow while maintaining strong detection capabilities through custom implementations.