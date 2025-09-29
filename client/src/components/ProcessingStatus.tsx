import { useState, useEffect } from "react";
import { Progress } from "@/components/ui/progress";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Eye, Brain, Search, CheckCircle } from "lucide-react";

interface ProcessingStep {
  id: string;
  label: string;
  status: 'pending' | 'processing' | 'completed';
  progress: number;
  icon: React.ReactNode;
}

interface ProcessingStatusProps {
  isProcessing: boolean;
  fileName?: string;
  progress?: number;
  message?: string;
  error?: string | null;
  onComplete?: () => void;
}

export default function ProcessingStatus({ 
  isProcessing, 
  fileName = "video.mp4", 
  progress = 0, 
  message = '',
  error = null,
  onComplete 
}: ProcessingStatusProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [steps, setSteps] = useState<ProcessingStep[]>([
    {
      id: 'upload',
      label: 'Uploading file',
      status: 'completed',
      progress: 100,
      icon: <CheckCircle className="w-4 h-4" />
    },
    {
      id: 'extract',
      label: 'Extracting frames',
      status: 'processing',
      progress: 0,
      icon: <Eye className="w-4 h-4" />
    },
    {
      id: 'detect',
      label: 'Detecting faces',
      status: 'pending',
      progress: 0,
      icon: <Search className="w-4 h-4" />
    },
    {
      id: 'analyze',
      label: 'AI analysis',
      status: 'pending',
      progress: 0,
      icon: <Brain className="w-4 h-4" />
    }
  ]);

  useEffect(() => {
    if (!isProcessing) return;

    const interval = setInterval(() => {
      setSteps(prevSteps => {
        const newSteps = [...prevSteps];
        const processingIndex = newSteps.findIndex(step => step.status === 'processing');
        
        if (processingIndex === -1) return newSteps;

        const currentProcessingStep = newSteps[processingIndex];
        const newProgress = Math.min(currentProcessingStep.progress + Math.random() * 15, 100);
        
        newSteps[processingIndex] = {
          ...currentProcessingStep,
          progress: newProgress
        };

        // Move to next step when current is complete
        if (newProgress >= 100) {
          newSteps[processingIndex].status = 'completed';
          if (processingIndex < newSteps.length - 1) {
            newSteps[processingIndex + 1].status = 'processing';
            setCurrentStep(processingIndex + 1);
          } else {
            // All steps completed
            setTimeout(() => onComplete?.(), 1000);
          }
        }

        return newSteps;
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isProcessing, onComplete]);

  // Use real progress from backend instead of simulated steps
  const overallProgress = progress;

  if (!isProcessing) {
    return null;
  }

  return (
    <Card className="p-6">
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center">
          <h3 className="text-lg font-medium mb-2">Analyzing {fileName}</h3>
          <p className="text-sm text-muted-foreground">
            {error ? error : message || 'Using lightweight ensemble models for deepfake detection'}
          </p>
        </div>

        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Overall Progress</span>
            <span className="font-mono">{Math.round(overallProgress)}%</span>
          </div>
          <Progress value={overallProgress} className="w-full" data-testid="progress-overall" />
        </div>

        {/* Processing Steps */}
        <div className="space-y-4">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center gap-4">
              <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                step.status === 'completed' 
                  ? 'bg-primary border-primary text-primary-foreground' 
                  : step.status === 'processing'
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-muted bg-muted/10 text-muted-foreground'
              }`}>
                {step.status === 'processing' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  step.icon
                )}
              </div>

              <div className="flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-medium ${
                    step.status === 'pending' ? 'text-muted-foreground' : ''
                  }`}>
                    {step.label}
                  </span>
                  <Badge variant={
                    step.status === 'completed' ? 'default' :
                    step.status === 'processing' ? 'secondary' : 'outline'
                  }>
                    {step.status === 'completed' ? 'Done' :
                     step.status === 'processing' ? 'Processing' : 'Waiting'}
                  </Badge>
                </div>
                
                {step.status !== 'pending' && (
                  <Progress 
                    value={step.progress} 
                    className="h-2" 
                    data-testid={`progress-${step.id}`}
                  />
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Technical Details */}
        <div className="p-4 bg-muted/20 rounded-lg">
          <p className="text-xs text-muted-foreground font-mono">
            Model: EfficientNet-B4 + Xception Ensemble • 
            Batch Size: 32 • 
            Confidence Threshold: 0.7
          </p>
        </div>
      </div>
    </Card>
  );
}