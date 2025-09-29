import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Shield, ShieldAlert, Download, Eye, AlertTriangle } from "lucide-react";

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

interface ResultsDisplayProps {
  result: AnalysisResult;
  onViewFrames?: () => void;
  onDownloadReport?: () => void;
}

export default function ResultsDisplay({ result, onViewFrames, onDownloadReport }: ResultsDisplayProps) {
  const getResultColor = () => {
    if (result.confidence < 0.5) return 'text-muted-foreground';
    return result.overall_prediction === 'FAKE' ? 'text-destructive' : 'text-primary';
  };

  const getResultIcon = () => {
    if (result.confidence < 0.5) {
      return <AlertTriangle className="w-8 h-8 text-muted-foreground" />;
    }
    return result.overall_prediction === 'FAKE' ? 
      <ShieldAlert className="w-8 h-8 text-destructive" /> : 
      <Shield className="w-8 h-8 text-primary" />;
  };

  const getResultText = () => {
    if (result.confidence < 0.5) return 'UNCERTAIN';
    return result.overall_prediction;
  };

  const getResultDescription = () => {
    if (result.confidence < 0.5) {
      return 'Analysis inconclusive. Consider uploading a higher quality file.';
    }
    if (result.overall_prediction === 'FAKE') {
      return 'AI-generated content detected. This media appears to be synthetically created.';
    }
    return 'Authentic content detected. This media appears to be genuine.';
  };

  const getConfidenceColor = () => {
    if (result.confidence < 0.5) return 'bg-muted';
    if (result.confidence < 0.7) return 'bg-chart-3'; // orange
    return result.overall_prediction === 'FAKE' ? 'bg-destructive' : 'bg-primary';
  };

  return (
    <Card className="p-6">
      <div className="space-y-6">
        {/* Main Result */}
        <div className="text-center">
          <div className="mb-4">
            {getResultIcon()}
          </div>
          
          <h2 className={`text-3xl font-bold mb-2 ${getResultColor()}`} data-testid="text-result">
            {getResultText()}
          </h2>
          
          <p className="text-muted-foreground mb-4">
            {getResultDescription()}
          </p>

          <Badge 
            variant={result.overall_prediction === 'FAKE' ? 'destructive' : 'default'}
            className="text-sm px-3 py-1"
            data-testid="badge-confidence"
          >
            {Math.round(result.confidence * 100)}% Confidence
          </Badge>
        </div>

        {/* Confidence Meter */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Detection Confidence</span>
            <span className="font-mono">{Math.round(result.confidence * 100)}%</span>
          </div>
          <Progress 
            value={result.confidence * 100} 
            className="w-full h-3"
            data-testid="progress-confidence"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>

        {/* File Details */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-muted/20 rounded-lg">
          <div>
            <p className="text-sm text-muted-foreground">File Name</p>
            <p className="font-mono text-sm truncate" data-testid="text-filename">{result.fileName}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Faces Detected</p>
            <p className="font-mono text-sm">{result.faces_detected}</p>
          </div>
          {result.frame_results && (
            <div>
              <p className="text-sm text-muted-foreground">Frames Analyzed</p>
              <p className="font-mono text-sm">{result.frame_results.length}</p>
            </div>
          )}
          <div>
            <p className="text-sm text-muted-foreground">Analysis Date</p>
            <p className="font-mono text-sm">{new Date(result.analysis_timestamp).toLocaleString()}</p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3">
          {result.fileType === 'video' && (
            <Button 
              variant="outline" 
              className="flex-1"
              onClick={() => {
                console.log('View frame analysis');
                onViewFrames?.();
              }}
              data-testid="button-view-frames"
            >
              <Eye className="w-4 h-4 mr-2" />
              View Frame Analysis
            </Button>
          )}
          
          <Button 
            variant="outline" 
            className="flex-1"
            onClick={() => {
              console.log('Download report');
              onDownloadReport?.();
            }}
            data-testid="button-download-report"
          >
            <Download className="w-4 h-4 mr-2" />
            Download Report
          </Button>
        </div>

        {/* Technical Details */}
        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            Technical Details
          </summary>
          <div className="mt-2 p-3 bg-muted/10 rounded text-xs font-mono space-y-1">
            <div>Model Architecture: Multi-CNN Ensemble</div>
            <div>Input Resolution: 224x224</div>
            <div>Preprocessing: Face alignment, normalization</div>
            <div>Threshold: 0.7 (adjustable)</div>
            <div>GPU Acceleration: Enabled</div>
          </div>
        </details>
      </div>
    </Card>
  );
}