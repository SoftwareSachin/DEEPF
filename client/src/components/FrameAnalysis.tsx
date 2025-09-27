import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Play, Pause, SkipBack, SkipForward, ZoomIn } from "lucide-react";

interface Frame {
  id: number;
  timestamp: number;
  confidence: number;
  isDeepfake: boolean;
  faces: FaceDetection[];
}

interface FaceDetection {
  id: string;
  bbox: { x: number; y: number; width: number; height: number };
  confidence: number;
  isDeepfake: boolean;
}

interface FrameAnalysisProps {
  frames: Frame[];
  videoUrl?: string;
  onClose?: () => void;
}

export default function FrameAnalysis({ frames, videoUrl, onClose }: FrameAnalysisProps) {
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedFace, setSelectedFace] = useState<string | null>(null);

  const currentFrame = frames[currentFrameIndex];
  const overallConfidence = frames.reduce((acc, frame) => acc + frame.confidence, 0) / frames.length;
  const fakeFrameCount = frames.filter(frame => frame.isDeepfake).length;

  const handlePrevFrame = () => {
    setCurrentFrameIndex(Math.max(0, currentFrameIndex - 1));
  };

  const handleNextFrame = () => {
    setCurrentFrameIndex(Math.min(frames.length - 1, currentFrameIndex + 1));
  };

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
    console.log(isPlaying ? 'Paused' : 'Playing');
  };

  const formatTimestamp = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceColor = (confidence: number, isDeepfake: boolean) => {
    if (confidence < 0.5) return 'text-muted-foreground';
    return isDeepfake ? 'text-destructive' : 'text-primary';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Frame-by-Frame Analysis</h2>
          <p className="text-muted-foreground">
            {frames.length} frames analyzed • {fakeFrameCount} potential deepfakes detected
          </p>
        </div>
        <Button variant="outline" onClick={onClose} data-testid="button-close-analysis">
          Close Analysis
        </Button>
      </div>

      {/* Overall Stats */}
      <Card className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-primary">{Math.round(overallConfidence * 100)}%</p>
            <p className="text-sm text-muted-foreground">Overall Confidence</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-destructive">{fakeFrameCount}</p>
            <p className="text-sm text-muted-foreground">Suspicious Frames</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold">{frames.length}</p>
            <p className="text-sm text-muted-foreground">Total Frames</p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Player */}
        <Card className="p-6">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Current Frame</h3>
            
            {/* Mock video frame */}
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center relative border">
              {videoUrl ? (
                <img src={videoUrl} alt="Video frame" className="w-full h-full object-cover rounded-lg" />
              ) : (
                <div className="text-center">
                  <Play className="w-16 h-16 text-muted-foreground mx-auto mb-2" />
                  <p className="text-muted-foreground">Frame {currentFrameIndex + 1}</p>
                </div>
              )}
              
              {/* Face detection overlays */}
              {currentFrame.faces.map((face) => (
                <div
                  key={face.id}
                  className={`absolute border-2 cursor-pointer ${
                    selectedFace === face.id 
                      ? 'border-primary' 
                      : face.isDeepfake 
                        ? 'border-destructive' 
                        : 'border-green-500'
                  }`}
                  style={{
                    left: `${face.bbox.x}%`,
                    top: `${face.bbox.y}%`,
                    width: `${face.bbox.width}%`,
                    height: `${face.bbox.height}%`,
                  }}
                  onClick={() => setSelectedFace(face.id)}
                  data-testid={`face-detection-${face.id}`}
                >
                  <Badge 
                    className="absolute -top-6 left-0 text-xs"
                    variant={face.isDeepfake ? "destructive" : "default"}
                  >
                    {Math.round(face.confidence * 100)}%
                  </Badge>
                </div>
              ))}
            </div>

            {/* Video Controls */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Button size="icon" variant="outline" onClick={handlePrevFrame} data-testid="button-prev-frame">
                  <SkipBack className="w-4 h-4" />
                </Button>
                <Button size="icon" onClick={handlePlayPause} data-testid="button-play-pause">
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </Button>
                <Button size="icon" variant="outline" onClick={handleNextFrame} data-testid="button-next-frame">
                  <SkipForward className="w-4 h-4" />
                </Button>
                <div className="flex-1 mx-4">
                  <Progress 
                    value={(currentFrameIndex / (frames.length - 1)) * 100} 
                    className="w-full"
                    data-testid="progress-timeline"
                  />
                </div>
                <Button size="icon" variant="outline" data-testid="button-zoom">
                  <ZoomIn className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Frame {currentFrameIndex + 1} of {frames.length}</span>
                <span>{formatTimestamp(currentFrame.timestamp)}</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Frame Details */}
        <Card className="p-6">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Frame Details</h3>
            
            {/* Current Frame Info */}
            <div className="p-4 bg-muted/20 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Detection Result</span>
                <Badge 
                  variant={currentFrame.isDeepfake ? "destructive" : "default"}
                  data-testid="badge-frame-result"
                >
                  {currentFrame.isDeepfake ? "FAKE" : "REAL"}
                </Badge>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Confidence</span>
                  <span className={`font-mono ${getConfidenceColor(currentFrame.confidence, currentFrame.isDeepfake)}`}>
                    {Math.round(currentFrame.confidence * 100)}%
                  </span>
                </div>
                <Progress 
                  value={currentFrame.confidence * 100} 
                  className="h-2"
                  data-testid="progress-frame-confidence"
                />
              </div>
            </div>

            {/* Face Detections */}
            <div>
              <h4 className="font-medium mb-3">Detected Faces ({currentFrame.faces.length})</h4>
              <div className="space-y-2">
                {currentFrame.faces.map((face) => (
                  <div 
                    key={face.id}
                    className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedFace === face.id 
                        ? 'border-primary bg-primary/5' 
                        : 'border-border hover:border-muted-foreground'
                    }`}
                    onClick={() => setSelectedFace(face.id)}
                    data-testid={`face-details-${face.id}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Face {face.id}</span>
                      <Badge 
                        variant={face.isDeepfake ? "destructive" : "default"}
                        className="text-xs"
                      >
                        {face.isDeepfake ? "FAKE" : "REAL"}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Confidence: {Math.round(face.confidence * 100)}%</span>
                      <span>
                        {Math.round(face.bbox.width)}×{Math.round(face.bbox.height)}px
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Frame Timeline */}
      <Card className="p-4">
        <h3 className="text-lg font-medium mb-4">Timeline</h3>
        <div className="flex gap-1 overflow-x-auto pb-2">
          {frames.map((frame, index) => (
            <button
              key={frame.id}
              className={`flex-shrink-0 w-12 h-8 rounded border text-xs font-mono transition-all ${
                index === currentFrameIndex
                  ? 'border-primary bg-primary text-primary-foreground'
                  : frame.isDeepfake
                    ? 'border-destructive bg-destructive/10 text-destructive'
                    : 'border-border bg-background hover:bg-muted'
              }`}
              onClick={() => setCurrentFrameIndex(index)}
              data-testid={`timeline-frame-${index}`}
            >
              {index + 1}
            </button>
          ))}
        </div>
      </Card>
    </div>
  );
}