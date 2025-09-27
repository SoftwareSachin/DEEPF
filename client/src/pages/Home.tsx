import { useState } from "react";
import FileUpload from "@/components/FileUpload";
import ProcessingStatus from "@/components/ProcessingStatus";
import ResultsDisplay from "@/components/ResultsDisplay";
import FrameAnalysis from "@/components/FrameAnalysis";
import { Button } from "@/components/ui/button";
import { Shield, ArrowLeft } from "lucide-react";

type AppState = 'upload' | 'processing' | 'results' | 'frameAnalysis';

interface AnalysisResult {
  isDeepfake: boolean;
  confidence: number;
  fileName: string;
  fileType: 'image' | 'video';
  processingTime: number;
  frameCount?: number;
  modelUsed: string[];
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

  // Mock frame data for demonstration
  const mockFrames: Frame[] = [
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

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setCurrentState('processing');
    setUploadProgress(0);

    // Simulate upload progress
    const uploadInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(uploadInterval);
          return 100;
        }
        return prev + Math.random() * 20;
      });
    }, 200);
  };

  const handleProcessingComplete = () => {
    // Simulate analysis result based on file name
    const fileName = selectedFile?.name || 'unknown.file';
    const isVideo = selectedFile?.type.startsWith('video/') || false;
    
    // Mock different results based on filename for demo
    const mockResult: AnalysisResult = fileName.toLowerCase().includes('fake') || fileName.toLowerCase().includes('deepfake') ? {
      isDeepfake: true,
      confidence: 0.87 + Math.random() * 0.1,
      fileName,
      fileType: isVideo ? 'video' : 'image',
      processingTime: 8.2 + Math.random() * 10,
      frameCount: isVideo ? 120 + Math.floor(Math.random() * 60) : undefined,
      modelUsed: ["EfficientNet-B4", "Xception"]
    } : {
      isDeepfake: false,
      confidence: 0.88 + Math.random() * 0.1,
      fileName,
      fileType: isVideo ? 'video' : 'image',
      processingTime: 5.1 + Math.random() * 8,
      frameCount: isVideo ? 150 + Math.floor(Math.random() * 80) : undefined,
      modelUsed: ["EfficientNet-B4", "ResNet-50"]
    };

    setAnalysisResult(mockResult);
    setCurrentState('results');
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
              onComplete={handleProcessingComplete}
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
            frames={mockFrames}
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