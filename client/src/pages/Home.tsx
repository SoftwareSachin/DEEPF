import { useState, useEffect } from "react";
import FileUpload from "@/components/FileUpload";
import ProcessingStatus from "@/components/ProcessingStatus";
import ResultsDisplay from "@/components/ResultsDisplay";
import FrameAnalysis from "@/components/FrameAnalysis";
import { Button } from "@/components/ui/button";
import { Shield, ArrowLeft } from "lucide-react";
import { io, Socket } from "socket.io-client";

type AppState = 'upload' | 'processing' | 'results' | 'frameAnalysis';

interface AnalysisResult {
  overall_prediction: string;
  confidence: number;
  fileName: string;
  fileType: 'image' | 'video';
  faces_detected: number;
  face_results?: Array<{
    face_id: number;
    bbox: number[];
    prediction: string;
    confidence: number;
  }>;
  frame_results?: Array<{
    frame_number: number;
    timestamp: number;
    faces_detected: number;
    face_results: Array<{
      face_id: number;
      bbox: number[];
      prediction: string;
      confidence: number;
    }>;
  }>;
  analysis_timestamp: string;
}

interface Frame {
  id: number;
  timestamp: number;
  confidence: number;
  isDeepfake: boolean;
  faces: Array<{
    id: string;
    bbox: { x: number; y: number; width: number; height: number };
    confidence: number;
    isDeepfake: boolean;
  }>;
}

export default function Home() {
  const [currentState, setCurrentState] = useState<AppState>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingMessage, setProcessingMessage] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io('http://127.0.0.1:5001');
    setSocket(newSocket);

    newSocket.on('progress', (data: { job_id: string; progress: number; message: string }) => {
      if (data.job_id === jobId) {
        setProcessingProgress(data.progress);
        setProcessingMessage(data.message);
        
        if (data.progress === 100) {
          // Processing complete, fetch results
          fetchResults(data.job_id);
        }
      }
    });

    return () => {
      newSocket.close();
    };
  }, [jobId]);

  // Note: mockFrames removed - now using real backend data via getFramesForAnalysis()
  const _unusedMockFrames: Frame[] = [
    {
      id: 1,
      timestamp: 0.5,
      confidence: 0.89,
      isDeepfake: true,
      faces: [
        {
          id: 'face-1',
          bbox: { x: 25, y: 20, width: 30, height: 40 },
          confidence: 0.89,
          isDeepfake: true
        }
      ]
    },
    {
      id: 2,
      timestamp: 1.0,
      confidence: 0.94,
      isDeepfake: false,
      faces: [
        {
          id: 'face-1',
          bbox: { x: 28, y: 22, width: 28, height: 38 },
          confidence: 0.94,
          isDeepfake: false
        }
      ]
    },
    {
      id: 3,
      timestamp: 1.5,
      confidence: 0.76,
      isDeepfake: true,
      faces: [
        {
          id: 'face-1',
          bbox: { x: 30, y: 25, width: 26, height: 35 },
          confidence: 0.76,
          isDeepfake: true
        }
      ]
    }
  ];

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    setCurrentState('processing');
    setUploadProgress(0);
    setProcessingProgress(0);
    setError(null);

    try {
      // Upload file to Python backend
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://127.0.0.1:5001/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setJobId(data.job_id);
      setUploadProgress(100);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setCurrentState('upload');
    }
  };

  const fetchResults = async (jobId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:5001/api/results/${jobId}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch results');
      }

      const data = await response.json();
      
      // Transform backend response to frontend format
      const result: AnalysisResult = {
        overall_prediction: data.results.overall_prediction,
        confidence: data.results.confidence,
        fileName: data.filename,
        fileType: data.file_type,
        faces_detected: data.results.faces_detected || 0,
        face_results: data.results.face_results,
        frame_results: data.results.frame_results,
        analysis_timestamp: data.results.analysis_timestamp
      };

      setAnalysisResult(result);
      setCurrentState('results');
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get results');
    }
  };

  // Transform frame results for frame analysis component
  const getFramesForAnalysis = (): Frame[] => {
    if (!analysisResult?.frame_results) return [];
    
    return analysisResult.frame_results.slice(0, 10).map((frameResult, index) => ({
      id: frameResult.frame_number,
      timestamp: frameResult.timestamp,
      confidence: frameResult.face_results.length > 0 ? 
        Math.max(...frameResult.face_results.map(f => f.confidence)) : 0,
      isDeepfake: frameResult.face_results.some(f => f.prediction === 'FAKE'),
      faces: frameResult.face_results.map(face => ({
        id: `face-${face.face_id}`,
        bbox: {
          x: face.bbox[0],
          y: face.bbox[1], 
          width: face.bbox[2] - face.bbox[0],
          height: face.bbox[3] - face.bbox[1]
        },
        confidence: face.confidence,
        isDeepfake: face.prediction === 'FAKE'
      }))
    }));
  };

  const handleViewFrames = () => {
    setCurrentState('frameAnalysis');
  };

  const handleDownloadReport = () => {
    console.log('Downloading report for:', analysisResult?.fileName);
    // In a real app, this would trigger a file download
  };

  const handleBackToResults = () => {
    setCurrentState('results');
  };

  const handleStartOver = () => {
    setCurrentState('upload');
    setSelectedFile(null);
    setAnalysisResult(null);
    setUploadProgress(0);
    setProcessingProgress(0);
    setProcessingMessage('');
    setJobId(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">Deepfake Detection</h1>
                <p className="text-sm text-muted-foreground">AI-powered media analysis</p>
              </div>
            </div>
            
            {currentState !== 'upload' && (
              <Button 
                variant="outline" 
                onClick={handleStartOver}
                data-testid="button-start-over"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Start Over
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {currentState === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <FileUpload 
              onFileSelect={handleFileSelect}
              isProcessing={false}
              uploadProgress={uploadProgress}
            />
          </div>
        )}

        {currentState === 'processing' && (
          <div className="max-w-2xl mx-auto">
            <ProcessingStatus
              isProcessing={true}
              fileName={selectedFile?.name}
              progress={processingProgress}
              message={processingMessage}
              error={error}
            />
          </div>
        )}

        {currentState === 'results' && analysisResult && (
          <div className="max-w-2xl mx-auto">
            <ResultsDisplay
              result={analysisResult}
              onViewFrames={analysisResult.fileType === 'video' ? handleViewFrames : undefined}
              onDownloadReport={handleDownloadReport}
            />
          </div>
        )}

        {currentState === 'frameAnalysis' && (
          <FrameAnalysis
            frames={getFramesForAnalysis()}
            onClose={handleBackToResults}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="text-center text-sm text-muted-foreground">
            <p>Powered by advanced CNN ensemble models for reliable deepfake detection</p>
            <p className="mt-1">EfficientNet-B4 • Xception • ResNet-50</p>
          </div>
        </div>
      </footer>
    </div>
  );
}